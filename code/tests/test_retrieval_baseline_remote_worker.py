from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from trec_rag.document_store import DocumentStore
from trec_rag.mixedbread_passage_scorer import MixedbreadPassageScorer
from trec_rag.retrieval_candidate_core import (
    CandidateCore,
    CandidateLaneStat,
    candidate_core_to_dict,
)
from trec_rag.retrieval_baseline_input_bundle import (
    export_portable_score_cache,
    seal_input_directory,
)
from trec_rag.retrieval_baseline_collection import collect_remote_scoring
from trec_rag.retrieval_baseline_remote_worker import (
    OFFICIAL_TOPIC_IDS,
    ordered_scoring_phases,
    run_canary_gated_scoring,
    run_remote_scoring,
)
from trec_rag.retrieval_baseline_runs import SemanticUnit, SourceDocument, TopicInput


def test_ordered_scoring_phases_requires_all_topics_and_fixed_canaries() -> None:
    canaries, remaining = ordered_scoring_phases(
        OFFICIAL_TOPIC_IDS,
        ("rag2026-1", "rag2026-18"),
    )

    assert canaries == ("rag2026-1", "rag2026-18")
    assert len(remaining) == 117
    assert not set(canaries).intersection(remaining)
    assert canaries + remaining != OFFICIAL_TOPIC_IDS
    assert set(canaries + remaining) == set(OFFICIAL_TOPIC_IDS)

    with pytest.raises(ValueError, match="119-topic"):
        ordered_scoring_phases(OFFICIAL_TOPIC_IDS[:-1], canaries)
    with pytest.raises(ValueError, match="canary"):
        ordered_scoring_phases(
            OFFICIAL_TOPIC_IDS,
            ("rag2026-18", "rag2026-1"),
        )


def test_canary_gate_runs_before_remaining_topics_with_one_score_callback() -> None:
    events: list[object] = []

    run_canary_gated_scoring(
        topic_ids=OFFICIAL_TOPIC_IDS,
        canary_topic_ids=("rag2026-1", "rag2026-18"),
        score_one=lambda topic_id: events.append(("score", topic_id)),
        verify_canaries=lambda topic_ids: events.append(("verify", topic_ids)),
    )

    assert events[:3] == [
        ("score", "rag2026-1"),
        ("score", "rag2026-18"),
        ("verify", ("rag2026-1", "rag2026-18")),
    ]
    assert len(events) == 120
    assert events[3] == ("score", "rag2026-0")
    assert events[-1] == ("score", "rag2026-118")


def test_failed_canary_verification_prevents_all_remaining_scoring() -> None:
    scored: list[str] = []

    def fail(_topic_ids: tuple[str, ...]) -> None:
        raise RuntimeError("replay mismatch")

    with pytest.raises(RuntimeError, match="replay mismatch"):
        run_canary_gated_scoring(
            topic_ids=OFFICIAL_TOPIC_IDS,
            canary_topic_ids=("rag2026-1", "rag2026-18"),
            score_one=scored.append,
            verify_canaries=fail,
        )

    assert scored == ["rag2026-1", "rag2026-18"]


class _Parameter:
    dtype = "torch.bfloat16"


class _Model:
    def __init__(self) -> None:
        self.calls = 0

    def parameters(self):
        return [_Parameter()]

    def predict(self, pairs, **_kwargs):
        self.calls += 1
        return [float(len(query) + len(passage)) for query, passage in pairs]


def _remote_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, object, _Model, dict[str, int], str]:
    input_dir = tmp_path / "input"
    source_run_id = "facet-deepseek-b40-v3"
    source_root = input_dir / "source" / source_run_id
    source_root.mkdir(parents=True)
    (source_root / "retrieval_export_manifest.json").write_text(
        json.dumps(
            {
                "export_code_commit": "a" * 40,
                "run_id": source_run_id,
                "selected_topic_ids": list(OFFICIAL_TOPIC_IDS),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    source_export_manifest_sha256 = sha256(
        (source_root / "retrieval_export_manifest.json").read_bytes()
    ).hexdigest()
    stored = DocumentStore(input_dir / "documents/v1").admit_text("shared passage")
    empty_admissions = sha256(b"").hexdigest()
    topic_stats: dict[str, dict[str, object]] = {}
    for topic_id in OFFICIAL_TOPIC_IDS:
        core = CandidateCore(
            topic_id=topic_id,
            lane_scores_sha256="c" * 64,
            candidate_docids=("doc",),
            pre_fallback_count=0,
            fallback_used=True,
            admission_multiplicity_histogram={},
            lanes=(
                CandidateLaneStat(
                    "original",
                    0.0,
                    0.0,
                    0.0,
                    "strictly_greater_than",
                    1,
                    0,
                    empty_admissions,
                ),
                CandidateLaneStat(
                    "facet:s1:text",
                    0.0,
                    0.0,
                    0.0,
                    "strictly_greater_than",
                    1,
                    0,
                    empty_admissions,
                ),
            ),
        )
        core_path = input_dir / "candidate-cores" / f"{topic_id}.json"
        core_path.parent.mkdir(parents=True, exist_ok=True)
        core_path.write_text(
            json.dumps(
                candidate_core_to_dict(core),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        topic_stats[topic_id] = {
            "cache_hits": 0,
            "cache_misses": 2,
            "candidate_core_sha256": sha256(core_path.read_bytes()).hexdigest(),
            "candidate_count": 1,
            "chunk_count": 1,
            "document_semantic_pair_count": 2,
            "first_seen_cache_key_count": 2,
            "passage_pair_count": 2,
            "reused_prior_topic_key_count": 0,
            "semantic_unit_count": 2,
            "unique_cache_key_count": 2,
        }
    portable_dir = input_dir / "portable-scores"
    portable_dir.mkdir()
    empty_cache = MixedbreadPassageScorer(
        tmp_path / "empty-cache", device="cpu", batch_size=32
    )
    empty_cache.score_cache.close()
    export_portable_score_cache(
        tmp_path / "empty-cache", portable_dir / "empty.jsonl"
    )
    seal_input_directory(
        input_dir,
        topic_ids=OFFICIAL_TOPIC_IDS,
        source_run_id=source_run_id,
        cache_stats={
            "candidate_documents": 119,
            "document_semantic_pairs": 238,
            "hits": 0,
            "misses": 238,
            "passage_pairs": 238,
            "portable_rows": 0,
            "unique_cache_keys": 238,
        },
        canary_topic_ids=("rag2026-1", "rag2026-18"),
        topic_stats=topic_stats,
    )

    def topic_loader(
        _source: Path,
        topic_id: str,
        _documents: Path,
        *,
        selected_docids: tuple[str, ...],
    ) -> TopicInput:
        assert selected_docids == ("doc",)
        return TopicInput(
            topic_id=topic_id,
            narrative=f"narrative {topic_id}",
            subnarratives=(SemanticUnit("s1", f"facet {topic_id}", ()),),
            documents=(
                SourceDocument(
                    "doc",
                    stored.content_sha256,
                    1,
                    {},
                    "shared passage",
                ),
            ),
            source_sha256s={
                "retrieval_export_manifest.json": source_export_manifest_sha256,
                "scoring/lane_scores.jsonl": "c" * 64,
            },
        )

    monkeypatch.setattr(
        "trec_rag.retrieval_baseline_remote_worker.load_topic_input",
        topic_loader,
    )
    monkeypatch.setattr(
        "trec_rag.retrieval_baseline_collection.load_topic_input",
        topic_loader,
    )
    source_revision = "b" * 40
    monkeypatch.setenv("TREC_RAG_SOURCE_REVISION", source_revision)
    model = _Model()
    counters = {"loader_calls": 0}

    def scorer_factory(root: Path, read_only: bool):
        def loader(**_kwargs):
            counters["loader_calls"] += 1
            if read_only:
                raise AssertionError("cache-only replay loaded the model")
            return model

        scorer = MixedbreadPassageScorer(
            root,
            device="cpu",
            model_loader=loader,
            batch_size=32,
            read_only=read_only,
        )
        scorer._device = "cuda"
        return scorer

    return input_dir, scorer_factory, model, counters, source_revision


def test_canary_replay_failure_writes_redacted_receipt_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir, scorer_factory, model, counters, source_revision = _remote_fixture(
        tmp_path, monkeypatch
    )

    def fail_compare(_first: Path, _second: Path) -> None:
        raise ValueError("forced secret-bearing mismatch detail")

    monkeypatch.setattr(
        "trec_rag.retrieval_baseline_remote_worker._compare_matrix_artifacts",
        fail_compare,
    )
    publication = tmp_path / "publication"
    with pytest.raises(ValueError, match="secret-bearing"):
        run_remote_scoring(
            input_dir=input_dir,
            work_root=tmp_path / "work",
            publication_dir=publication,
            scorer_factory=scorer_factory,
        )

    receipt_path = publication / "remote-scoring-failure-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt == {
        "canary_topic_ids": ["rag2026-1", "rag2026-18"],
        "failure_stage": "canary_replay",
        "failure_type": "ValueError",
        "input_manifest_sha256": sha256(
            (input_dir / "input-manifest.json").read_bytes()
        ).hexdigest(),
        "schema_version": "retrieval-baseline-remote-failure-v1",
        "scored_topic_ids": ["rag2026-1", "rag2026-18"],
        "source_revision": source_revision,
        "status": "failed",
    }
    assert sorted(path.name for path in (publication / "matrices").iterdir()) == [
        "rag2026-1",
        "rag2026-18",
    ]
    assert model.calls == 4
    assert counters["loader_calls"] == 1
    assert not (publication / "portable-scores").exists()
    assert not (publication / "runs").exists()
    assert b"secret-bearing" not in receipt_path.read_bytes()
    assert all(
        b"shared passage" not in path.read_bytes()
        for path in publication.rglob("*")
        if path.is_file()
    )


def test_remote_scoring_uses_one_live_model_and_fresh_cache_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir, scorer_factory, model, counters, source_revision = _remote_fixture(
        tmp_path, monkeypatch
    )

    receipt = run_remote_scoring(
        input_dir=input_dir,
        work_root=tmp_path / "work",
        publication_dir=tmp_path / "publication",
        scorer_factory=scorer_factory,
    )

    assert receipt["status"] == "complete"
    assert receipt["topic_count"] == 119
    assert receipt["portable_cache_row_count"] == 238
    assert counters["loader_calls"] == 1
    assert model.calls == 238
    publication = tmp_path / "publication"
    assert len(list((publication / "matrices").glob("rag2026-*"))) == 119
    assert len(list((publication / "replay-receipts/final").glob("*.json"))) == 119
    assert (
        publication / "runs/narrative/r_output_trec_rag_2026.tsv"
    ).read_text(encoding="utf-8").count("\n") == 119

    sums = publication / "SHA256SUMS"
    lines = []
    for path in sorted(
        (path for path in publication.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(publication).as_posix(),
    ):
        relative = path.relative_to(publication).as_posix()
        lines.append(f"{sha256(path.read_bytes()).hexdigest()}  ./{relative}\n")
    sums.write_text("".join(lines), encoding="utf-8")
    input_manifest_sha256 = sha256(
        (input_dir / "input-manifest.json").read_bytes()
    ).hexdigest()
    publication_manifest = {
        "input_manifest_sha256": input_manifest_sha256,
        "remote_scoring_receipt_sha256": sha256(
            (publication / "remote-scoring-receipt.json").read_bytes()
        ).hexdigest(),
        "schema_version": "retrieval-baseline-publication-v1",
        "sha256s_sha256": sha256(sums.read_bytes()).hexdigest(),
        "source_revision": source_revision,
        "status": "complete",
        "task_name": "candidate-core-all",
    }
    (publication / "publication-manifest.json").write_text(
        json.dumps(publication_manifest, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )

    collected = collect_remote_scoring(
        publication_dir=publication,
        input_dir=input_dir,
        expected_input_manifest_sha256=input_manifest_sha256,
        expected_source_revision=source_revision,
        shared_cache_root=tmp_path / "shared-cache",
        work_root=tmp_path / "collection-work",
        output_dir=tmp_path / "collection-output",
        scorer_factory=lambda root: scorer_factory(root, True),
    )

    assert collected["status"] == "complete"
    assert collected["fresh_replay"]["model_batches"] == 0
    assert collected["shared_cache_final_replay"]["model_batches"] == 0
    merge = json.loads(
        (tmp_path / "collection-output/merge-receipt.json").read_text()
    )
    assert merge["before_row_count"] == 0
    assert merge["after_row_count"] == merge["inserted_count"] == 238
