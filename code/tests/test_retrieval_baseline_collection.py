from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from trec_rag.mixedbread_passage_scorer import MixedbreadPassageScorer
from trec_rag.retrieval_baseline_collection import (
    merge_portable_scores,
    verify_publication_closure,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = REPO_ROOT / "code/tools/collect_retrieval_baseline_worker.sh"


def _publication_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "publication"
    root.mkdir(parents=True)
    artifact = root / "nested/artifact.json"
    artifact.parent.mkdir()
    artifact.write_text('{"ok":true}\n', encoding="utf-8")
    sums = root / "SHA256SUMS"
    sums.write_text(
        f"{sha256(artifact.read_bytes()).hexdigest()}  ./nested/artifact.json\n",
        encoding="utf-8",
    )
    input_digest = "a" * 64
    source_revision = "b" * 40
    manifest = {
        "input_manifest_sha256": input_digest,
        "remote_scoring_receipt_sha256": "c" * 64,
        "schema_version": "retrieval-baseline-publication-v1",
        "sha256s_sha256": sha256(sums.read_bytes()).hexdigest(),
        "source_revision": source_revision,
        "status": "complete",
        "task_name": "candidate-core-all",
    }
    (root / "publication-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return root, input_digest, source_revision


def test_publication_closure_accepts_exact_manifest_last_file_set(tmp_path: Path) -> None:
    root, input_digest, source_revision = _publication_fixture(tmp_path)

    verified = verify_publication_closure(
        root,
        expected_input_manifest_sha256=input_digest,
        expected_source_revision=source_revision,
    )

    assert verified["status"] == "complete"
    assert verified["input_manifest_sha256"] == input_digest


def test_publication_closure_rejects_tamper_extra_files_and_wrong_identity(
    tmp_path: Path,
) -> None:
    root, input_digest, source_revision = _publication_fixture(tmp_path)
    (root / "nested/artifact.json").write_text('{"ok":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256SUMS|digest"):
        verify_publication_closure(
            root,
            expected_input_manifest_sha256=input_digest,
            expected_source_revision=source_revision,
        )

    root, input_digest, source_revision = _publication_fixture(tmp_path / "extra")
    (root / "undeclared.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        verify_publication_closure(
            root,
            expected_input_manifest_sha256=input_digest,
            expected_source_revision=source_revision,
        )

    root, input_digest, source_revision = _publication_fixture(tmp_path / "identity")
    with pytest.raises(ValueError, match="input manifest identity"):
        verify_publication_closure(
            root,
            expected_input_manifest_sha256="d" * 64,
            expected_source_revision=source_revision,
        )


def _portable_cache(
    root: Path,
    destination: Path,
    rows: list[tuple[str, str, float]],
) -> Path:
    scorer = MixedbreadPassageScorer(root, device="cpu", batch_size=32)
    try:
        scorer.score_cache.add_many(rows)
        scorer.score_cache.export_portable_jsonl(destination)
    finally:
        scorer.score_cache.close()
    return destination


def test_merge_portable_scores_is_locked_transactional_and_idempotent(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    seed = MixedbreadPassageScorer(shared, device="cpu", batch_size=32)
    try:
        seed.score_cache.add_many([("q1", "p1", 1.0)])
    finally:
        seed.score_cache.close()
    portable = _portable_cache(
        tmp_path / "source",
        tmp_path / "scores.jsonl",
        [("q1", "p1", 1.0), ("q2", "p2", 2.0)],
    )

    first = merge_portable_scores(portable, shared)
    second = merge_portable_scores(portable, shared)

    assert first["before_row_count"] == 1
    assert first["after_row_count"] == 2
    assert first["inserted_count"] == 1
    assert first["already_identical_count"] == 1
    assert first["conflict_count"] == 0
    assert second["before_row_count"] == second["after_row_count"] == 2
    assert second["inserted_count"] == 0
    assert second["receipt_reused"] is True
    assert Path(first["lock_path"]).name == ".retrieval-baseline-merge.lock"


def test_merge_conflict_rolls_back_every_new_score(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    seed = MixedbreadPassageScorer(shared, device="cpu", batch_size=32)
    try:
        seed.score_cache.add_many([("q1", "p1", 1.0)])
    finally:
        seed.score_cache.close()
    conflicting = _portable_cache(
        tmp_path / "conflicting-source",
        tmp_path / "conflicting.jsonl",
        [("q1", "p1", 9.0), ("q3", "p3", 3.0)],
    )

    with pytest.raises(ValueError, match="conflicting score"):
        merge_portable_scores(conflicting, shared)

    check = MixedbreadPassageScorer(
        shared, device="cpu", batch_size=32, read_only=True
    )
    try:
        assert check.score_cache.lookup_many([("q1", "p1"), ("q3", "p3")]) == [
            1.0,
            None,
        ]
    finally:
        check.score_cache.close()


def test_collection_wrapper_is_private_explicit_and_merges_only_after_download() -> None:
    source = COLLECTOR.read_text(encoding="utf-8")

    privacy = source.index('buckets info "$bucket"')
    download = source.index('buckets sync "$publication_prefix"')
    collect = source.index("retrieval_baseline_collection")
    assert privacy < download < collect
    assert "--shared-cache" in source
    assert "--input-manifest-sha256" in source
    assert "--source-revision" in source
    assert "--no-delete" in source
    assert "publication-manifest.json" in source
