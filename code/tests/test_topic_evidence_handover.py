"""Contracts for the deterministic Topic 213 evidence handover inputs."""

from __future__ import annotations

import hashlib
from itertools import pairwise
import json
from pathlib import Path

import pytest

import trec_rag.topic_evidence_handover as handover_module
from trec_rag.remote_client import RemotePyseriniThrottled
from trec_rag.topic_evidence_handover import (
    CANONICAL_SUB_NARRATIVES,
    EligibleDocument,
    TopicEvidenceInputs,
    fetch_missing_documents,
    load_topic213_inputs,
    missing_document_ids,
    render_handover_markdown,
    score_sub_narrative_pairs,
    validate_reviewed_handover,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDOVER_DIRECTORY = (
    REPO_ROOT / "reports/experiments/rag25_topic213_evidence_handover_v1"
)
HANDOVER = HANDOVER_DIRECTORY / "handover.json"
MARKDOWN = HANDOVER_DIRECTORY / "handover.md"
MANIFEST = HANDOVER_DIRECTORY / "manifest.yaml"
CANONICAL_QRELS = (
    REPO_ROOT
    / "trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels"
    / "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
)


def canonical_eligible_grades() -> dict[str, int]:
    """Return the released Topic 213 grade-2-through-4 provenance mapping."""

    grades: dict[str, int] = {}
    for line in CANONICAL_QRELS.read_text(encoding="utf-8").splitlines():
        topic_id, _iteration, document_id, raw_grade = line.split()
        grade = int(raw_grade)
        if topic_id == "213" and grade in {2, 3, 4}:
            grades[document_id] = grade
    assert len(grades) == 173
    return grades


@pytest.fixture
def fixture_paths(tmp_path: Path) -> dict[str, object]:
    """Small real-format inputs; canonical population validation is disabled."""

    topic_tsv = tmp_path / "topics.tsv"
    topic_tsv.write_text(
        "212\tAnother topic\n213\tThe authoritative Topic 213 narrative\n",
        encoding="utf-8",
    )
    nuggets_jsonl = tmp_path / "nuggets.jsonl"
    nuggets_jsonl.write_text(
        json.dumps(
            {
                "qid": "213",
                "nuggets": [
                    {"mapped_sub_narrative": f"SN{number}"}
                    for number in range(1, 11)
                ]
                + [{"mapped_sub_narrative": "SN1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.txt"
    qrels.write_text(
        "213 0 doc-1 2\n213 0 doc-2 3\n213 0 doc-3 4\n"
        "213 0 excluded-1 1\n212 0 excluded-2 4\n",
        encoding="utf-8",
    )
    accepted_union = tmp_path / "accepted_union.jsonl"
    accepted_union.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"topic_id": "213", "document_id": "doc-2", "text": "second text"},
                {"topic_id": "213", "document_id": "doc-1", "text": "first text"},
                {"topic_id": "212", "document_id": "excluded-2", "text": "other topic"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    supplemental_documents = tmp_path / "supplemental.jsonl"
    supplemental_documents.write_text(
        json.dumps({"document_id": "doc-3", "text": "third text"}) + "\n",
        encoding="utf-8",
    )
    return {
        "topic_tsv": topic_tsv,
        "nuggets_jsonl": nuggets_jsonl,
        "qrels": qrels,
        "accepted_union": accepted_union,
        "supplemental_documents": supplemental_documents,
        "canonical": False,
    }


@pytest.fixture
def sample_handover() -> dict[str, object]:
    return {
        "topic_id": "213",
        "narrative": "The authoritative Topic 213 narrative",
        "sub_narratives": [
            {
                "sub_narrative": sub_narrative,
                "documents": [
                    {
                        "document_id": f"doc-{(number - 1) * 5 + rank}",
                        "topic_qrel_grade": 2,
                        "support_score": 2,
                        "claims": [f"Supported claim {number}-{rank}"],
                    }
                    for rank in range(1, 6)
                ],
            }
            for number, sub_narrative in enumerate(CANONICAL_SUB_NARRATIVES, start=1)
        ],
    }


@pytest.fixture
def sample_inputs() -> TopicEvidenceInputs:
    return TopicEvidenceInputs(
        topic_id="213",
        narrative="The authoritative Topic 213 narrative",
        sub_narratives=CANONICAL_SUB_NARRATIVES,
        documents=tuple(
            EligibleDocument(
                document_id=f"doc-{index}",
                text=f"source text {index}",
                topic_qrel_grade=2,
            )
            for index in range(1, 51)
        ),
    )


def deterministic_scorer(_sub_narrative: str, chunk_texts: list[str]) -> list[float]:
    """A local score function whose expected order comes from fixture IDs."""

    return [float(int(chunk_text.rsplit(" ", 1)[-1])) for chunk_text in chunk_texts]


@pytest.fixture
def canonical_fetch_paths(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    topic_tsv = tmp_path / "topics.tsv"
    topic_tsv.write_text("213\tCanonical narrative\n", encoding="utf-8")
    nuggets_jsonl = tmp_path / "nuggets.jsonl"
    nuggets_jsonl.write_text(
        json.dumps(
            {
                "qid": "213",
                "nuggets": [
                    {"mapped_sub_narrative": label}
                    for label in CANONICAL_SUB_NARRATIVES
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.txt"
    qrels.write_text(
        "".join(f"213 0 doc-{index:04d} 2\n" for index in range(1, 174)),
        encoding="utf-8",
    )
    accepted_union = tmp_path / "accepted.jsonl"
    accepted_union.write_text(
        "".join(
            json.dumps(
                {
                    "topic_id": "213",
                    "document_id": f"doc-{index:04d}",
                    "text": f"accepted text {index}",
                }
            )
            + "\n"
            for index in range(1, 173)
        ),
        encoding="utf-8",
    )
    mapping_bytes = "".join(
        f"doc-{index:04d}\t2\n" for index in range(1, 174)
    ).encode("utf-8")
    monkeypatch.setattr(
        handover_module,
        "CANONICAL_ELIGIBLE_MAPPING_SHA256",
        hashlib.sha256(mapping_bytes).hexdigest(),
        raising=False,
    )
    monkeypatch.setattr(
        handover_module,
        "CANONICAL_INITIAL_MISSING_SHA256",
        hashlib.sha256(b"doc-0173\n").hexdigest(),
        raising=False,
    )
    return {
        "topic_tsv": topic_tsv,
        "nuggets_jsonl": nuggets_jsonl,
        "qrels": qrels,
        "accepted_union": accepted_union,
        "supplemental_documents": tmp_path / "supplemental.jsonl",
    }


class _Response:
    def __init__(
        self,
        raw: bytes,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = raw
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requests: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs):
        self.requests.append((url, kwargs))
        return self.response


def _install_remote(
    monkeypatch,
    session: _Session,
    *,
    token: str = "private-test-token",
) -> None:
    monkeypatch.setenv(
        "INDEX_URL",
        "https://example.test/v1/climbmix-400b/search",
    )
    monkeypatch.setenv("PYSERINI_API_TOKEN", token)
    monkeypatch.setattr(handover_module, "load_repo_env", lambda _root: None)
    monkeypatch.setattr(
        handover_module,
        "rate_limited_session",
        lambda _config: session,
    )


def test_fetch_missing_requires_nonblank_token_before_request(
    canonical_fetch_paths, monkeypatch
):
    session = _Session(_Response(b'{"contents":"unused"}'))
    _install_remote(monkeypatch, session, token="   ")

    with pytest.raises(ValueError, match="PYSERINI_API_TOKEN"):
        fetch_missing_documents(**canonical_fetch_paths)

    assert session.requests == []


def test_fetch_missing_uses_document_endpoint_and_writes_normalized_hashed_record(
    canonical_fetch_paths, monkeypatch
):
    raw = b'{"contents":"  normalized\\n  document   text  "}'
    session = _Session(_Response(raw))
    _install_remote(monkeypatch, session)

    result = fetch_missing_documents(**canonical_fetch_paths)

    assert result == {"fetched_count": 1, "remaining_missing_count": 0}
    assert session.requests[0][0] == (
        "https://example.test/v1/climbmix-400b/doc/doc-0173"
    )
    assert session.requests[0][1]["headers"]["Authorization"] == (
        "Bearer private-test-token"
    )
    record_text = canonical_fetch_paths["supplemental_documents"].read_text(
        encoding="utf-8"
    )
    assert "private-test-token" not in record_text
    assert json.loads(record_text) == {
        "document_id": "doc-0173",
        "text": "normalized document text",
        "response_sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_fetch_missing_complete_cache_returns_before_env_session_or_output_open(
    canonical_fetch_paths, monkeypatch
):
    canonical_fetch_paths["supplemental_documents"].write_text(
        json.dumps(
            {
                "document_id": "doc-0173",
                "text": "complete text",
                "response_sha256": "c" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lock_entries: list[str] = []
    real_file_lock = handover_module.FileLock

    def recording_lock(path):
        lock_entries.append(str(path))
        return real_file_lock(path)

    monkeypatch.setattr(handover_module, "FileLock", recording_lock)
    monkeypatch.setattr(
        handover_module,
        "load_repo_env",
        lambda _root: pytest.fail("complete resume loaded environment"),
    )
    monkeypatch.setattr(
        handover_module,
        "rate_limited_session",
        lambda _config: pytest.fail("complete resume created a session"),
    )

    assert fetch_missing_documents(**canonical_fetch_paths) == {
        "fetched_count": 0,
        "remaining_missing_count": 0,
    }
    assert lock_entries == [
        str(canonical_fetch_paths["supplemental_documents"]) + ".lock"
    ]


def test_fetch_missing_partial_resume_requests_only_uncached_id(
    canonical_fetch_paths, monkeypatch
):
    accepted = canonical_fetch_paths["accepted_union"]
    accepted.write_text(
        "".join(accepted.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]),
        encoding="utf-8",
    )
    canonical_fetch_paths["supplemental_documents"].write_text(
        json.dumps(
            {
                "document_id": "doc-0172",
                "text": "cached supplemental text",
                "response_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        handover_module,
        "CANONICAL_INITIAL_MISSING_SHA256",
        hashlib.sha256(b"doc-0172\ndoc-0173\n").hexdigest(),
    )
    session = _Session(_Response(b'{"contents":"last text"}'))
    _install_remote(monkeypatch, session)

    result = fetch_missing_documents(**canonical_fetch_paths)

    assert result == {"fetched_count": 1, "remaining_missing_count": 0}
    assert [request[0].rsplit("/", 1)[-1] for request in session.requests] == [
        "doc-0173"
    ]


def test_fetch_missing_locks_before_first_supplemental_read(
    canonical_fetch_paths, monkeypatch
):
    canonical_fetch_paths["supplemental_documents"].write_text(
        '{"document_id":',
        encoding="utf-8",
    )
    lock_entries: list[str] = []

    class CompletingWriterLock:
        def __init__(self, path):
            lock_entries.append(str(path))

        def __enter__(self):
            canonical_fetch_paths["supplemental_documents"].write_text(
                json.dumps(
                    {
                        "document_id": "doc-0173",
                        "text": "concurrent process text",
                        "response_sha256": "b" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(handover_module, "FileLock", CompletingWriterLock)
    monkeypatch.setattr(
        handover_module,
        "load_repo_env",
        lambda _root: pytest.fail("lock recheck loaded environment"),
    )

    assert fetch_missing_documents(**canonical_fetch_paths) == {
        "fetched_count": 0,
        "remaining_missing_count": 0,
    }
    assert lock_entries == [
        str(canonical_fetch_paths["supplemental_documents"]) + ".lock"
    ]


@pytest.mark.parametrize("invalid_source", ("qrels", "sub_narratives"))
def test_fetch_missing_validates_canonical_inputs_before_network(
    canonical_fetch_paths, monkeypatch, invalid_source
):
    if invalid_source == "qrels":
        qrels = canonical_fetch_paths["qrels"]
        qrels.write_text(
            "".join(qrels.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]),
            encoding="utf-8",
        )
        expected = "173"
    else:
        nuggets = json.loads(
            canonical_fetch_paths["nuggets_jsonl"].read_text(encoding="utf-8")
        )
        nuggets["nuggets"][0]["mapped_sub_narrative"] = "altered"
        canonical_fetch_paths["nuggets_jsonl"].write_text(
            json.dumps(nuggets) + "\n",
            encoding="utf-8",
        )
        expected = "exact released"
    monkeypatch.setattr(
        handover_module,
        "load_repo_env",
        lambda _root: pytest.fail("invalid canonical source loaded environment"),
    )

    with pytest.raises(ValueError, match=expected):
        fetch_missing_documents(**canonical_fetch_paths)


@pytest.mark.parametrize("substitution", ("eligible_mapping", "accepted_union"))
def test_fetch_missing_rejects_same_count_identity_substitution(
    canonical_fetch_paths, monkeypatch, substitution
):
    if substitution == "eligible_mapping":
        qrels = canonical_fetch_paths["qrels"]
        lines = qrels.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[-1] = "213 0 doc-9999 2\n"
        qrels.write_text("".join(lines), encoding="utf-8")
        expected = "eligible mapping identity"
    else:
        accepted = canonical_fetch_paths["accepted_union"]
        rows = [
            json.loads(line)
            for line in accepted.read_text(encoding="utf-8").splitlines()
        ]
        rows[-1]["document_id"] = "doc-0173"
        accepted.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        expected = "initial missing identity"
    monkeypatch.setattr(
        handover_module,
        "load_repo_env",
        lambda _root: pytest.fail("substituted identity loaded environment"),
    )

    with pytest.raises(ValueError, match=expected):
        fetch_missing_documents(**canonical_fetch_paths)


def test_fetch_missing_429_preserves_retry_after_without_retry(
    canonical_fetch_paths, monkeypatch
):
    session = _Session(
        _Response(
            b'{"error":"throttled"}',
            status_code=429,
            headers={"Retry-After": "7"},
        )
    )
    _install_remote(monkeypatch, session)

    with pytest.raises(RemotePyseriniThrottled) as caught:
        fetch_missing_documents(**canonical_fetch_paths)

    assert caught.value.retry_after_seconds == 7.0
    assert len(session.requests) == 1


def test_missing_document_ids_returns_sorted_qrel_documents_without_text(sample_inputs):
    """Catches a fetch plan that omits or nondeterministically orders absent qrel IDs."""

    assert missing_document_ids(
        eligible_docids={"doc-a", "doc-b", "doc-c"},
        available_docids={"doc-b"},
    ) == ["doc-a", "doc-c"]


def test_shortlist_scores_every_pair_and_orders_descending(sample_inputs):
    """Catches a shortlist that skips pairs or ranks lower model scores first."""

    result = score_sub_narrative_pairs(
        sample_inputs,
        deterministic_scorer,
        shortlist_depth=2,
    )

    assert result["pair_count"] == len(sample_inputs.sub_narratives) * len(sample_inputs.documents)
    assert all(
        row["model_score"] >= next_row["model_score"]
        for rows in result["shortlists"].values()
        for row, next_row in pairwise(rows)
    )


def test_shortlist_keeps_qrel_grade_separate_from_model_score(sample_inputs):
    """Catches a review shortlist that overwrites organizer grades with model logits."""

    row = score_sub_narrative_pairs(sample_inputs, deterministic_scorer)["shortlists"][
        sample_inputs.sub_narratives[0]
    ][0]

    assert isinstance(row["topic_qrel_grade"], int)
    assert isinstance(row["model_score"], float)


def test_load_topic213_inputs_preserves_population_and_subnarratives(fixture_paths):
    """Catches a loader that includes ineligible docs or alters nugget labels."""

    loaded = load_topic213_inputs(**fixture_paths)

    assert loaded.topic_id == "213"
    assert loaded.narrative == "The authoritative Topic 213 narrative"
    assert loaded.sub_narratives == tuple(f"SN{number}" for number in range(1, 11))
    assert [(document.document_id, document.topic_qrel_grade) for document in loaded.documents] == [
        ("doc-1", 2),
        ("doc-2", 3),
        ("doc-3", 4),
    ]


def test_loader_rejects_missing_text_for_eligible_document(fixture_paths):
    """Catches a loader that silently scores an eligible document without text."""

    accepted_union = fixture_paths["accepted_union"]
    assert isinstance(accepted_union, Path)
    accepted_union.write_text(
        json.dumps({"topic_id": "213", "document_id": "doc-1", "text": "first text"})
        + "\n"
        + json.dumps({"topic_id": "213", "document_id": "doc-2", "text": ""})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing text"):
        load_topic213_inputs(**fixture_paths)


def test_loader_rejects_duplicate_accepted_union_document_id(fixture_paths):
    """Catches ambiguous joins caused by duplicate accepted-union identities."""

    accepted_union = fixture_paths["accepted_union"]
    assert isinstance(accepted_union, Path)
    accepted_union.write_text(
        "\n".join(
            (
                json.dumps({"topic_id": "213", "document_id": "doc-1", "text": "first text"}),
                json.dumps({"topic_id": "213", "document_id": "doc-1", "text": "duplicate text"}),
                json.dumps({"topic_id": "213", "document_id": "doc-2", "text": "second text"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate document ID"):
        load_topic213_inputs(**fixture_paths)


def test_loader_merges_supplemental_document_text_before_eligibility_validation(fixture_paths):
    """Catches a loader that rejects qrel documents absent from the accepted union."""

    loaded = load_topic213_inputs(**fixture_paths)

    assert loaded.documents[-1].document_id == "doc-3"
    assert loaded.documents[-1].text == "third text"


def test_loader_allows_matching_accepted_union_and_supplemental_text(fixture_paths):
    """Catches a merge that rejects a harmless duplicate source record."""

    supplemental_documents = fixture_paths["supplemental_documents"]
    assert isinstance(supplemental_documents, Path)
    supplemental_documents.write_text(
        "\n".join(
            (
                json.dumps({"document_id": "doc-1", "text": "first text"}),
                json.dumps({"document_id": "doc-3", "text": "third text"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_topic213_inputs(**fixture_paths)

    assert [document.document_id for document in loaded.documents] == ["doc-1", "doc-2", "doc-3"]


def test_loader_rejects_conflicting_accepted_union_and_supplemental_text(fixture_paths):
    """Catches a merge that silently selects one source's conflicting text."""

    supplemental_documents = fixture_paths["supplemental_documents"]
    assert isinstance(supplemental_documents, Path)
    supplemental_documents.write_text(
        "\n".join(
            (
                json.dumps({"document_id": "doc-1", "text": "conflicting text"}),
                json.dumps({"document_id": "doc-3", "text": "third text"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting text"):
        load_topic213_inputs(**fixture_paths)


def test_loader_requires_canonical_population_of_173(fixture_paths):
    """Catches accidental use of a partial population in canonical artifact mode."""

    fixture_paths["canonical"] = True

    with pytest.raises(ValueError, match="173"):
        load_topic213_inputs(**fixture_paths)


def test_canonical_loader_requires_the_released_sub_narrative_tuple(fixture_paths):
    """Catches a canonical load that accepts ten altered or reordered labels."""

    qrels = fixture_paths["qrels"]
    accepted_union = fixture_paths["accepted_union"]
    nuggets_jsonl = fixture_paths["nuggets_jsonl"]
    assert isinstance(qrels, Path)
    assert isinstance(accepted_union, Path)
    assert isinstance(nuggets_jsonl, Path)
    qrels.write_text(
        "".join(f"213 0 doc-{index} 2\n" for index in range(1, 174)),
        encoding="utf-8",
    )
    accepted_union.write_text(
        "".join(
            json.dumps({"topic_id": "213", "document_id": f"doc-{index}", "text": f"text {index}"})
            + "\n"
            for index in range(1, 174)
        ),
        encoding="utf-8",
    )
    nuggets_jsonl.write_text(
        json.dumps(
            {
                "qid": "213",
                "nuggets": [
                    {"mapped_sub_narrative": label}
                    for label in (*CANONICAL_SUB_NARRATIVES[1:], CANONICAL_SUB_NARRATIVES[0])
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fixture_paths["supplemental_documents"] = None
    fixture_paths["canonical"] = True

    with pytest.raises(ValueError, match="exact released mapped_sub_narrative"):
        load_topic213_inputs(**fixture_paths)


def test_handover_requires_five_supported_unique_documents(sample_handover, sample_inputs):
    """Catches a reviewed record with a merely related, unusable document."""

    sample_handover["sub_narratives"][0]["documents"][0]["support_score"] = 1

    with pytest.raises(ValueError, match="support_score"):
        validate_reviewed_handover(
            sample_handover,
            inputs=sample_inputs,
        )


def test_handover_rejects_duplicate_or_uneligible_documents(sample_handover, sample_inputs):
    """Catches five-item lists that are not five eligible, distinct documents."""

    sample_handover["sub_narratives"][0]["documents"][4]["document_id"] = "doc-1"

    with pytest.raises(ValueError, match="unique"):
        validate_reviewed_handover(
            sample_handover,
            inputs=sample_inputs,
        )


def test_handover_requires_the_exact_canonical_sub_narrative_tuple(sample_handover, sample_inputs):
    """Catches a final artifact that swaps the released coverage-area order."""

    rows = sample_handover["sub_narratives"]
    assert isinstance(rows, list)
    rows[0]["sub_narrative"], rows[1]["sub_narrative"] = (
        rows[1]["sub_narrative"],
        rows[0]["sub_narrative"],
    )

    with pytest.raises(ValueError, match="exact released mapped_sub_narrative"):
        validate_reviewed_handover(sample_handover, inputs=sample_inputs)


def test_handover_requires_qrel_grade_to_match_input_provenance(sample_handover, sample_inputs):
    """Catches a reviewer record that alters the organizer qrel grade."""

    rows = sample_handover["sub_narratives"]
    assert isinstance(rows, list)
    rows[0]["documents"][0]["topic_qrel_grade"] = 3

    with pytest.raises(ValueError, match="does not match input provenance"):
        validate_reviewed_handover(sample_handover, inputs=sample_inputs)


def test_handover_accepts_explicit_qrel_grade_mapping(sample_handover):
    """Catches a validator that only works with the loader's concrete type."""

    validate_reviewed_handover(
        sample_handover,
        qrel_grades={f"doc-{index}": 2 for index in range(1, 51)},
    )


def test_handover_rejects_unallowlisted_content_and_absolute_paths(sample_handover, sample_inputs):
    """Catches artifacts that add raw content fields or absolute machine paths."""

    sample_handover["sub_narratives"][0]["documents"][0]["content"] = "raw document body"

    with pytest.raises(ValueError, match="allowlisted"):
        validate_reviewed_handover(sample_handover, inputs=sample_inputs)

    del sample_handover["sub_narratives"][0]["documents"][0]["content"]
    sample_handover["sub_narratives"][0]["documents"][0]["claims"] = ["evidence=/tmp/raw.txt"]

    with pytest.raises(ValueError, match="absolute filesystem path"):
        validate_reviewed_handover(sample_handover, inputs=sample_inputs)


@pytest.mark.parametrize(
    ("target", "unsafe_value", "error"),
    (
        ("narrative", "PYSERINI_API_TOKEN=private-token", "sensitive credential value"),
        ("claims", "Authorization: Bearer private-token", "sensitive credential value"),
        ("claims", r"evidence=C:\private\raw.txt", "absolute filesystem path"),
        ("review_rationale", "evidence=~/private/raw.txt", "absolute filesystem path"),
    ),
)
def test_handover_rejects_sensitive_values_and_embedded_paths(
    sample_handover,
    sample_inputs,
    target,
    unsafe_value,
    error,
):
    """Catches secrets and filesystem paths hidden in permitted string fields."""

    document = sample_handover["sub_narratives"][0]["documents"][0]
    if target == "narrative":
        sample_handover["narrative"] = unsafe_value
    elif target == "claims":
        document["claims"] = [unsafe_value]
    else:
        document["review_rationale"] = unsafe_value

    with pytest.raises(ValueError, match=error):
        validate_reviewed_handover(sample_handover, inputs=sample_inputs)


def test_render_handover_markdown_exposes_review_fields_without_document_text(sample_handover):
    """Catches a renderer that leaks source text instead of the reviewed claims."""

    sample_handover["sub_narratives"][0]["documents"][0]["text"] = "raw ClimbMix source text"

    markdown = render_handover_markdown(sample_handover)

    assert CANONICAL_SUB_NARRATIVES[0] in markdown
    assert "doc-1" in markdown
    assert "topic qrel grade: 2" in markdown
    assert "support score: 2" in markdown
    assert "Supported claim 1-1" in markdown
    assert "raw ClimbMix source text" not in markdown


def test_canonical_handover_has_ten_by_five_supported_documents():
    """Catches a tracked handover with incomplete or ineligible evidence coverage."""

    payload = json.loads(HANDOVER.read_text(encoding="utf-8"))

    validate_reviewed_handover(
        payload,
        qrel_grades=canonical_eligible_grades(),
    )
    assert len(payload["sub_narratives"]) == 10
    assert all(len(row["documents"]) == 5 for row in payload["sub_narratives"])


def test_canonical_handover_is_sanitized_and_deterministically_rendered():
    """Catches raw text, local paths, credentials, or renderer drift in tracked files."""

    payload = json.loads(HANDOVER.read_text(encoding="utf-8"))
    handover_text = HANDOVER.read_text(encoding="utf-8")
    markdown_text = MARKDOWN.read_text(encoding="utf-8")
    manifest_text = MANIFEST.read_text(encoding="utf-8")
    combined = handover_text + markdown_text + manifest_text

    assert "/home/" not in combined
    assert "PYSERINI_API_TOKEN" not in combined
    assert '"text":' not in handover_text
    assert markdown_text == render_handover_markdown(payload)
