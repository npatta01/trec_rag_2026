import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time

import pytest

import trec_rag.rerank_cache_promotion as promotion_module
from trec_rag.pipeline_config import load_pipeline_config
from trec_rag.rerank_cache_promotion import (
    BundleExpectations,
    CacheBundlePaths,
    CacheBundleValidationError,
    ExpectedDocumentIdentity,
    ExpectedWindowIdentity,
    _score_contexts_from_config,
    promote_cache_bundle,
    validate_cache_bundle,
)
from trec_rag.rerank_score_cache import GlobalScoreCache, ScoreCacheContext


@dataclass(frozen=True)
class _FixtureBundle:
    paths: CacheBundlePaths
    expectations: BundleExpectations
    document_cache_path: Path
    window_cache_path: Path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_bundle(tmp_path: Path) -> _FixtureBundle:
    document_context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        backend_version="5.6.0",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        model_revision="revision-a",
        score_representation="raw_logits",
        inference_dtype="bfloat16",
        input_policy="trec_rag_raw_v2",
        max_length=8,
        requested_max_length=10,
        pair_buffer_tokens=2,
        score_kind="doc_max_10_buf2",
    )
    window_context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        backend_version="5.6.0",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        model_revision="revision-a",
        score_representation="raw_logits",
        inference_dtype="bfloat16",
        input_policy="trec_rag_raw_v2",
        max_length=4,
        requested_max_length=4,
        pair_buffer_tokens=0,
        score_kind="window",
        chunk_max_characters=100,
        chunk_overlap_characters=10,
    )
    score_cache_root = tmp_path / "staged" / "score_cache"
    document_cache = GlobalScoreCache(score_cache_root, document_context)
    window_cache = GlobalScoreCache(score_cache_root, window_context)
    document_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    document_inputs: list[tuple[str, str, float]] = []
    window_inputs: list[tuple[str, str, float]] = []
    expected_documents: dict[tuple[str, str], ExpectedDocumentIdentity] = {}
    expected_windows: dict[tuple[str, str, int], ExpectedWindowIdentity] = {}
    topic_window_counts: dict[str, int] = {}

    for topic_number, topic_id in enumerate(("1", "2"), start=1):
        query_text = f"query for topic {topic_id}"
        query_hash = _sha256_text(query_text)
        topic_window_counts[topic_id] = 0
        for rank in (1, 2):
            docid = f"doc-{topic_id}-{rank}"
            document_text = f"document text for {docid}"
            document_hash = _sha256_text(document_text)
            document_score = float(topic_number * 10 + rank)
            document_inputs.append((query_text, document_text, document_score))
            document_rows.append(
                {
                    "topic_id": topic_id,
                    "docid": docid,
                    "rank": rank,
                    "score": document_score,
                    **document_context.artifact_metadata,
                    "query_sha256": query_hash,
                    "text_sha256": document_hash,
                    "score_cache_key": document_cache.cache_key(
                        query_text=query_text, text=document_text
                    ),
                }
            )
            expected_documents[(topic_id, docid)] = ExpectedDocumentIdentity(
                rank=rank,
                query_sha256=query_hash,
                text_sha256=document_hash,
            )
            chunk_count = rank
            for chunk_index in range(chunk_count):
                chunk_text = f"window {chunk_index} for {docid}"
                chunk_hash = _sha256_text(chunk_text)
                start_char = chunk_index * 10
                end_char = start_char + len(chunk_text)
                window_score = float(topic_number * 100 + rank * 10 + chunk_index)
                window_inputs.append((query_text, chunk_text, window_score))
                window_rows.append(
                    {
                        "topic_id": topic_id,
                        "docid": docid,
                        "rank": rank,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "chunk_id": f"{docid}:{chunk_index:04d}",
                        "start_char": start_char,
                        "end_char": end_char,
                        "score": window_score,
                        **window_context.artifact_metadata,
                        "query_sha256": query_hash,
                        "document_text_sha256": document_hash,
                        "text_sha256": chunk_hash,
                        "score_cache_key": window_cache.cache_key(
                            query_text=query_text, text=chunk_text
                        ),
                    }
                )
                expected_windows[(topic_id, docid, chunk_index)] = (
                    ExpectedWindowIdentity(
                        chunk_count=chunk_count,
                        chunk_id=f"{docid}:{chunk_index:04d}",
                        start_char=start_char,
                        end_char=end_char,
                        query_sha256=query_hash,
                        document_text_sha256=document_hash,
                        text_sha256=chunk_hash,
                    )
                )
                topic_window_counts[topic_id] += 1

    document_cache.add_many(document_inputs)
    window_cache.add_many(window_inputs)
    document_cache.close()
    window_cache.close()
    document_artifact = tmp_path / "staged" / "results" / "document.jsonl"
    window_artifact = tmp_path / "staged" / "results" / "window.jsonl"
    _write_jsonl(document_artifact, document_rows)
    _write_jsonl(window_artifact, window_rows)
    return _FixtureBundle(
        paths=CacheBundlePaths(
            document_artifact=document_artifact,
            window_artifact=window_artifact,
            score_cache_root=score_cache_root,
        ),
        expectations=BundleExpectations(
            topic_ids=("1", "2"),
            documents_per_topic=2,
            window_rows_per_topic=topic_window_counts,
            documents=expected_documents,
            windows=expected_windows,
            document_context=document_context,
            window_context=window_context,
        ),
        document_cache_path=document_cache.path,
        window_cache_path=window_cache.path,
    )


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _mutate_score_cache(path: Path, statement: str, parameters=()) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(statement, parameters)
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for sidecar in (Path(f"{path}-wal"), Path(f"{path}-shm")):
        sidecar.unlink(missing_ok=True)


def _seed_destination_cache(path_root: Path, context: ScoreCacheContext) -> GlobalScoreCache:
    cache = GlobalScoreCache(path_root, context)
    cache.add_many([("old query", "old text", 0.25)])
    cache.close()
    return cache


def _live_destination(
    tmp_path: Path,
    bundle: _FixtureBundle,
    *,
    document_context: ScoreCacheContext | None = None,
    window_context: ScoreCacheContext | None = None,
) -> tuple[CacheBundlePaths, Path, Path]:
    document_context = document_context or bundle.expectations.document_context
    window_context = window_context or bundle.expectations.window_context
    assert document_context is not None
    assert window_context is not None
    destination_root = tmp_path / "live" / "score_cache"
    destination = CacheBundlePaths(
        document_artifact=tmp_path / "live" / "results" / "document.jsonl",
        window_artifact=tmp_path / "live" / "results" / "window.jsonl",
        score_cache_root=destination_root,
    )
    return (
        destination,
        destination_root.joinpath(*document_context.path_parts),
        destination_root.joinpath(*window_context.path_parts),
    )


def _write_runtime_status(
    bundle: _FixtureBundle,
    *,
    state: str = "completed",
) -> Path:
    document_context = bundle.expectations.document_context
    window_context = bundle.expectations.window_context
    assert document_context is not None
    assert window_context is not None
    payload = {
        "state": state,
        "run_id": "test-modal-run",
        "input_sha256": "f" * 64,
        "document_rows": 4,
        "window_rows": 6,
        "document_validation": {
            "rows": 4,
            "rows_by_topic": {"1": 2, "2": 2},
            "sha256": _sha256_file(bundle.paths.document_artifact),
        },
        "window_validation": {
            "rows": 6,
            "rows_by_topic": {"1": 3, "2": 3},
            "sha256": _sha256_file(bundle.paths.window_artifact),
        },
        "scoring_contract": {
            "artifact_schema_version": 2,
            "backend": document_context.backend,
            "backend_version": document_context.backend_version,
            "model": document_context.model,
            "model_revision": document_context.model_revision,
            "score_representation": document_context.score_representation,
            "inference_dtype": document_context.inference_dtype,
            "input_policy": document_context.input_policy,
            "candidate_limit": 2,
            "topics": ["1", "2"],
            "document_max_length": document_context.requested_max_length,
            "document_pair_buffer_tokens": document_context.pair_buffer_tokens,
            "document_model_max_length": document_context.max_length,
            "window_max_length": window_context.max_length,
            "chunk_max_characters": window_context.chunk_max_characters,
            "chunk_overlap_characters": window_context.chunk_overlap_characters,
        },
    }
    path = bundle.paths.document_artifact.parent / "runtime_status.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_validates_exact_artifact_and_global_cache_coverage(tmp_path):
    bundle = _build_bundle(tmp_path)

    result = validate_cache_bundle(bundle.paths, expectations=bundle.expectations)

    assert result.reconciled_to_local_inputs is True
    assert result.matched_selected_context is True
    assert result.topic_document_counts == {"1": 2, "2": 2}
    assert result.topic_window_counts == {"1": 3, "2": 3}
    assert result.document_artifact.row_count == 4
    assert result.window_artifact.row_count == 6
    assert result.cache_extra_keys == {"document": 0, "window": 0}


def test_validates_completed_modal_runtime_status(tmp_path):
    bundle = _build_bundle(tmp_path)
    status_path = _write_runtime_status(bundle)

    result = validate_cache_bundle(
        bundle.paths,
        expectations=bundle.expectations,
        runtime_status_path=status_path,
    )

    assert result.runtime_status is not None
    assert result.runtime_status.payload["state"] == "completed"
    assert result.runtime_status.sha256 == _sha256_file(status_path)


def test_validates_exact_legacy_top_level_runtime_status_shape(tmp_path):
    bundle = _build_bundle(tmp_path)
    payload = {
        "state": "completed",
        "document_rows": 4,
        "window_rows": 6,
        "document_sha256": _sha256_file(bundle.paths.document_artifact),
        "window_sha256": _sha256_file(bundle.paths.window_artifact),
    }
    assert "scoring_contract" not in payload
    status_path = bundle.paths.document_artifact.parent / "legacy_runtime_status.json"
    status_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_cache_bundle(
        bundle.paths,
        expectations=bundle.expectations,
        runtime_status_path=status_path,
    )

    assert result.runtime_status is not None
    assert result.runtime_status.payload == payload


def test_rejects_incomplete_modal_runtime_status(tmp_path):
    bundle = _build_bundle(tmp_path)
    status_path = _write_runtime_status(bundle, state="running")

    with pytest.raises(CacheBundleValidationError, match="state must be 'completed'"):
        validate_cache_bundle(
            bundle.paths,
            expectations=bundle.expectations,
            runtime_status_path=status_path,
        )


def test_rejects_completed_runtime_status_without_artifact_digests(tmp_path):
    bundle = _build_bundle(tmp_path)
    status_path = _write_runtime_status(bundle)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["document_validation"].pop("sha256")
    payload["window_validation"].pop("sha256")
    status_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        CacheBundleValidationError,
        match="must include a document artifact SHA-256",
    ):
        validate_cache_bundle(
            bundle.paths,
            expectations=bundle.expectations,
            runtime_status_path=status_path,
        )


def test_rejects_runtime_status_with_partial_scoring_contract(tmp_path):
    bundle = _build_bundle(tmp_path)
    status_path = _write_runtime_status(bundle)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["scoring_contract"].pop("topics")
    status_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        CacheBundleValidationError,
        match="scoring_contract is missing required fields: topics",
    ):
        validate_cache_bundle(
            bundle.paths,
            expectations=bundle.expectations,
            runtime_status_path=status_path,
        )


@pytest.mark.parametrize(
    ("field_path", "bad_value", "error"),
    [
        (("document_rows",), 3, "document_rows=3; expected 4"),
        (
            ("window_validation", "sha256"),
            "0" * 64,
            "window_validation.sha256 does not match",
        ),
    ],
)
def test_rejects_modal_runtime_status_that_differs_from_artifacts(
    tmp_path, field_path, bad_value, error
):
    bundle = _build_bundle(tmp_path)
    status_path = _write_runtime_status(bundle)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    target = payload
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = bad_value
    status_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CacheBundleValidationError, match=error):
        validate_cache_bundle(
            bundle.paths,
            expectations=bundle.expectations,
            runtime_status_path=status_path,
        )


def test_rejects_artifact_cache_key_that_does_not_match_content_hash(tmp_path):
    bundle = _build_bundle(tmp_path)
    rows = _read_rows(bundle.paths.document_artifact)
    rows[0]["text_sha256"] = "a" * 64
    _write_jsonl(bundle.paths.document_artifact, rows)

    with pytest.raises(
        CacheBundleValidationError, match="score_cache_key does not match"
    ):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


def test_rejects_score_cache_context_metadata_conflict(tmp_path):
    bundle = _build_bundle(tmp_path)
    _mutate_score_cache(
        bundle.document_cache_path,
        "UPDATE cache_meta SET value = ? WHERE key = 'context_sha256'",
        ("0" * 64,),
    )

    with pytest.raises(CacheBundleValidationError, match="metadata|context"):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


def test_rejects_model_context_that_differs_from_selected_config(tmp_path):
    bundle = _build_bundle(tmp_path)
    expected_document_context = replace(
        bundle.expectations.document_context,
        model="different/model",
    )
    expected_window_context = replace(
        bundle.expectations.window_context,
        model="different/model",
    )
    expectations = replace(
        bundle.expectations,
        document_context=expected_document_context,
        window_context=expected_window_context,
    )

    with pytest.raises(
        CacheBundleValidationError,
        match="document artifact context differs.*model=",
    ):
        validate_cache_bundle(bundle.paths, expectations=expectations)


def test_rejects_chunk_policy_that_differs_from_selected_config(tmp_path):
    bundle = _build_bundle(tmp_path)
    expectations = replace(
        bundle.expectations,
        window_context=replace(
            bundle.expectations.window_context,
            chunk_overlap_characters=11,
        ),
    )

    with pytest.raises(
        CacheBundleValidationError,
        match="window artifact context differs.*chunk_overlap_characters=10.*config: 11",
    ):
        validate_cache_bundle(bundle.paths, expectations=expectations)


def test_selected_config_context_uses_score_builder_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: topics.tsv, format: tsv}
query_understanding: {variants: [{name: original, type: original_topic}]}
retrievers: [{name: bm25, type: pyserini_remote, query_variants: [original], hits: 1000}]
ranking:
  type: coverage_aware_long_doc_aggregate
  reranker:
    model: mixedbread-ai/mxbai-rerank-base-v2
    score_source: cached_artifacts
    document_score_path: document.jsonl
    window_score_path: window.jsonl
    formula: {}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
        encoding="utf-8",
    )

    document_context, window_context = _score_contexts_from_config(
        load_pipeline_config(config_path)
    )

    assert document_context.model_revision == (
        "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
    )
    assert document_context.backend_version == "5.6.0"
    assert document_context.score_representation == "raw_logits"
    assert document_context.inference_dtype == "bfloat16"
    assert document_context.input_policy == "trec_rag_raw_v2"
    assert document_context.requested_max_length == 32768
    assert document_context.max_length == 32256
    assert document_context.pair_buffer_tokens == 512
    assert window_context.max_length == 1024
    assert window_context.chunk_max_characters == 3500
    assert window_context.chunk_overlap_characters == 350


def test_rejects_incomplete_window_coverage(tmp_path):
    bundle = _build_bundle(tmp_path)
    rows = _read_rows(bundle.paths.window_artifact)
    _write_jsonl(bundle.paths.window_artifact, rows[:-1])

    with pytest.raises(CacheBundleValidationError, match="incomplete window coverage"):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


def test_rejects_conflicting_score_in_global_cache(tmp_path):
    bundle = _build_bundle(tmp_path)
    _mutate_score_cache(
        bundle.document_cache_path,
        "UPDATE scores SET score = score + 1.0 WHERE rowid = 1",
    )

    with pytest.raises(CacheBundleValidationError, match="global cache conflicts"):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


def test_validates_sqlite_score_row_and_count(tmp_path):
    bundle = _build_bundle(tmp_path)

    result = validate_cache_bundle(bundle.paths, expectations=bundle.expectations)

    assert result.document_score_cache.row_count == 4
    assert result.document_score_cache.unique_key_count == 4


def test_rejects_nonfinite_raw_logit(tmp_path):
    bundle = _build_bundle(tmp_path)
    rows = _read_rows(bundle.paths.window_artifact)
    rows[0]["score"] = float("inf")
    _write_jsonl(bundle.paths.window_artifact, rows)

    with pytest.raises(
        CacheBundleValidationError, match="raw-logit score must be finite"
    ):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


def test_rejects_boolean_artifact_score(tmp_path):
    bundle = _build_bundle(tmp_path)
    rows = _read_rows(bundle.paths.window_artifact)
    rows[0]["score"] = True
    _write_jsonl(bundle.paths.window_artifact, rows)

    with pytest.raises(CacheBundleValidationError, match="JSON number"):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


@pytest.mark.parametrize("sidecar", ["-wal", "-shm"])
def test_rejects_unsealed_score_cache_sidecars(tmp_path, sidecar):
    bundle = _build_bundle(tmp_path)
    sidecar_path = Path(f"{bundle.document_cache_path}{sidecar}")
    sidecar_path.write_bytes(b"stale")

    with pytest.raises(CacheBundleValidationError, match="sidecar|WAL"):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


@pytest.mark.parametrize(
    "ddl",
    [
        "CREATE TRIGGER extra_trigger AFTER INSERT ON scores BEGIN SELECT 1; END",
        "CREATE VIEW extra_view AS SELECT 1",
        "CREATE TABLE extra_table (value TEXT)",
        "CREATE INDEX extra_index ON scores(score)",
    ],
)
def test_rejects_unsupported_sqlite_objects(tmp_path, ddl):
    bundle = _build_bundle(tmp_path)
    _mutate_score_cache(bundle.document_cache_path, ddl)

    with pytest.raises(CacheBundleValidationError, match="schema|object"):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


def test_rejects_malformed_sqlite_metadata(tmp_path):
    bundle = _build_bundle(tmp_path)
    _mutate_score_cache(
        bundle.document_cache_path,
        "UPDATE cache_meta SET value = ? WHERE key = 'context_json'",
        ("not-json",),
    )

    with pytest.raises(CacheBundleValidationError, match="metadata|context"):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


def test_rejects_missing_score_coverage(tmp_path):
    bundle = _build_bundle(tmp_path)
    _mutate_score_cache(
        bundle.document_cache_path,
        "DELETE FROM scores WHERE rowid = 1",
    )

    with pytest.raises(CacheBundleValidationError, match="missing"):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


def test_rejects_nonfinite_score_stored_in_sqlite(tmp_path):
    bundle = _build_bundle(tmp_path)
    with sqlite3.connect(bundle.document_cache_path) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE scores SET score = ? WHERE rowid = 1", (float("inf"),)
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    Path(f"{bundle.document_cache_path}-wal").unlink(missing_ok=True)
    Path(f"{bundle.document_cache_path}-shm").unlink(missing_ok=True)

    with pytest.raises(CacheBundleValidationError, match="finite"):
        validate_cache_bundle(bundle.paths, expectations=bundle.expectations)


def test_promotion_rejects_global_cache_keys_not_used_by_artifacts(tmp_path):
    bundle = _build_bundle(tmp_path)
    document_context = bundle.expectations.document_context
    assert document_context is not None
    extra_cache = GlobalScoreCache(bundle.paths.score_cache_root, document_context)
    extra_cache.add_many([("unreferenced query", "unreferenced document", 1.25)])
    extra_cache.close()
    validation = validate_cache_bundle(bundle.paths, expectations=bundle.expectations)
    assert validation.cache_extra_keys == {"document": 1, "window": 0}
    destination, _, _ = _live_destination(tmp_path, bundle)

    with pytest.raises(
        CacheBundleValidationError,
        match="exact artifact/cache key coverage.*'document': 1",
    ):
        promote_cache_bundle(
            bundle.paths,
            destination,
            archive_root=tmp_path / "archives",
            expectations=bundle.expectations,
        )

    assert not (tmp_path / "archives").exists()


def test_promotes_validated_files_and_archives_previous_files(tmp_path):
    bundle = _build_bundle(tmp_path)
    runtime_status_path = _write_runtime_status(bundle)
    reviewed_runtime_payload = json.loads(
        runtime_status_path.read_text(encoding="utf-8")
    )
    destination, destination_document_cache, destination_window_cache = (
        _live_destination(tmp_path, bundle)
    )
    old_files = {
        destination.document_artifact: b"old document artifact\n",
        destination.window_artifact: b"old window artifact\n",
    }
    for path, content in old_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _seed_destination_cache(tmp_path / "live" / "score_cache", bundle.expectations.document_context)
    _seed_destination_cache(tmp_path / "live" / "score_cache", bundle.expectations.window_context)

    result = promote_cache_bundle(
        bundle.paths,
        destination,
        archive_root=tmp_path / "archives",
        expectations=bundle.expectations,
        runtime_status_path=runtime_status_path,
    )

    assert destination.document_artifact.read_bytes() == (
        bundle.paths.document_artifact.read_bytes()
    )
    assert (
        destination.window_artifact.read_bytes()
        == bundle.paths.window_artifact.read_bytes()
    )
    document_destination = GlobalScoreCache(
        tmp_path / "live" / "score_cache", bundle.expectations.document_context
    )
    window_destination = GlobalScoreCache(
        tmp_path / "live" / "score_cache", bundle.expectations.window_context
    )
    assert document_destination.lookup_many([("old query", "old text")])[0] == 0.25
    assert window_destination.lookup_many([("query for topic 1", "window 0 for doc-1-1")])[0] == 110.0
    document_destination.close()
    window_destination.close()
    promotion_manifest = json.loads(
        (result.archive_dir / "promotion_manifest.json").read_text(encoding="utf-8")
    )
    assert promotion_manifest["reviewed_modal_runtime_status"]["payload"] == (
        reviewed_runtime_payload
    )
    assert set(result.backups) == {"document_artifact", "window_artifact"}
    for label, backup_path in result.backups.items():
        assert backup_path.read_bytes() == old_files[result.destinations[label]]


def test_keyboard_interrupt_mid_swap_rolls_back_entire_bundle(tmp_path, monkeypatch):
    bundle = _build_bundle(tmp_path)
    destination, destination_document_cache, destination_window_cache = (
        _live_destination(tmp_path, bundle)
    )
    old_files = {
        destination.document_artifact: b"old document artifact\n",
        destination.window_artifact: b"old window artifact\n",
    }
    for path, content in old_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _seed_destination_cache(tmp_path / "live" / "score_cache", bundle.expectations.document_context)
    _seed_destination_cache(tmp_path / "live" / "score_cache", bundle.expectations.window_context)

    real_replace = promotion_module._atomic_replace
    incoming_replacements = 0
    interrupted = False

    def interrupt_second_incoming(source: Path, target: Path) -> None:
        nonlocal incoming_replacements, interrupted
        if source.name.endswith(".incoming"):
            incoming_replacements += 1
            if incoming_replacements == 2 and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("injected mid-swap interrupt")
        real_replace(source, target)

    monkeypatch.setattr(promotion_module, "_atomic_replace", interrupt_second_incoming)

    with pytest.raises(KeyboardInterrupt, match="injected mid-swap interrupt"):
        promote_cache_bundle(
            bundle.paths,
            destination,
            archive_root=tmp_path / "archives",
            expectations=bundle.expectations,
        )

    assert incoming_replacements == 2
    for path, expected in old_files.items():
        assert path.read_bytes() == expected
    destination_cache = GlobalScoreCache(
        tmp_path / "live" / "score_cache", bundle.expectations.document_context
    )
    assert destination_cache.lookup_many(
        [("query for topic 1", "document text for doc-1-1")]
    )[0] is None
    assert destination_cache.connection.execute(
        "SELECT count(*) FROM imports"
    ).fetchone()[0] == 0
    destination_cache.close()
    transaction_files = [
        path
        for path in (tmp_path / "live").rglob("*")
        if path.is_file()
        and (path.name.endswith(".incoming") or path.name.endswith(".rollback"))
    ]
    assert transaction_files == []


def test_late_source_change_rolls_back_authenticated_import(tmp_path, monkeypatch):
    bundle = _build_bundle(tmp_path)
    destination, _, _ = _live_destination(tmp_path, bundle)
    real_sha256_file = promotion_module._sha256_file
    document_hash_calls = 0

    def changed_after_import(path: Path) -> str:
        nonlocal document_hash_calls
        if Path(path) == bundle.document_cache_path:
            document_hash_calls += 1
            if document_hash_calls >= 3:
                return "f" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(
        promotion_module,
        "_sha256_file",
        changed_after_import,
    )

    with pytest.raises(
        CacheBundleValidationError,
        match="sealed score cache changed",
    ):
        promote_cache_bundle(
            bundle.paths,
            destination,
            archive_root=tmp_path / "archives",
            expectations=bundle.expectations,
        )

    destination_cache = GlobalScoreCache(
        destination.score_cache_root,
        bundle.expectations.document_context,
    )
    assert destination_cache.lookup_many(
        [("query for topic 1", "document text for doc-1-1")]
    )[0] is None
    assert destination_cache.connection.execute(
        "SELECT count(*) FROM imports"
    ).fetchone()[0] == 0
    destination_cache.close()
    assert not (tmp_path / "archives").exists()


def test_source_wal_created_during_import_is_rejected_and_rolled_back(
    tmp_path,
    monkeypatch,
):
    bundle = _build_bundle(tmp_path)
    destination, _, _ = _live_destination(tmp_path, bundle)
    real_import_scores = GlobalScoreCache.import_scores
    writers: list[sqlite3.Connection] = []
    injected = False

    def import_then_open_source_wal(cache, source, **kwargs):
        nonlocal injected
        receipt = real_import_scores(cache, source, **kwargs)
        if not injected:
            injected = True
            writer = sqlite3.connect(bundle.document_cache_path)
            writer.execute("UPDATE scores SET score = score + 1000 WHERE rowid = 1")
            writer.commit()
            writers.append(writer)
            assert Path(f"{bundle.document_cache_path}-wal").exists()
        return receipt

    monkeypatch.setattr(
        GlobalScoreCache,
        "import_scores",
        import_then_open_source_wal,
    )

    try:
        with pytest.raises(
            CacheBundleValidationError,
            match="sidecar|sealed",
        ):
            promote_cache_bundle(
                bundle.paths,
                destination,
                archive_root=tmp_path / "archives",
                expectations=bundle.expectations,
            )
    finally:
        for writer in writers:
            writer.close()

    destination_cache = GlobalScoreCache(
        destination.score_cache_root,
        bundle.expectations.document_context,
    )
    assert destination_cache.lookup_many(
        [("query for topic 1", "document text for doc-1-1")]
    )[0] is None
    assert destination_cache.connection.execute(
        "SELECT count(*) FROM imports"
    ).fetchone()[0] == 0
    destination_cache.close()


def test_failure_before_manifest_replace_rolls_back(tmp_path, monkeypatch):
    bundle = _build_bundle(tmp_path)
    destination, _, _ = _live_destination(tmp_path, bundle)
    old_document = b"old document artifact\n"
    old_window = b"old window artifact\n"
    destination.document_artifact.parent.mkdir(parents=True, exist_ok=True)
    destination.document_artifact.write_bytes(old_document)
    destination.window_artifact.write_bytes(old_window)

    def fail_before_manifest_replace(path: Path, payload, **_kwargs) -> None:
        raise OSError("injected failure before manifest replacement")

    monkeypatch.setattr(
        promotion_module,
        "_write_json_durable",
        fail_before_manifest_replace,
    )

    with pytest.raises(OSError, match="before manifest replacement"):
        promote_cache_bundle(
            bundle.paths,
            destination,
            archive_root=tmp_path / "archives",
            expectations=bundle.expectations,
        )

    assert destination.document_artifact.read_bytes() == old_document
    assert destination.window_artifact.read_bytes() == old_window
    assert list((tmp_path / "archives").glob("*/promotion_manifest.json")) == []
    destination_cache = GlobalScoreCache(
        destination.score_cache_root,
        bundle.expectations.document_context,
    )
    assert destination_cache.lookup_many(
        [("query for topic 1", "document text for doc-1-1")]
    )[0] is None
    destination_cache.close()


def test_failure_after_manifest_replace_preserves_committed_promotion(
    tmp_path,
    monkeypatch,
):
    bundle = _build_bundle(tmp_path)
    destination, _, _ = _live_destination(tmp_path, bundle)
    old_document = b"old document artifact\n"
    old_window = b"old window artifact\n"
    destination.document_artifact.parent.mkdir(parents=True, exist_ok=True)
    destination.document_artifact.write_bytes(old_document)
    destination.window_artifact.write_bytes(old_window)
    real_write_json_durable = promotion_module._write_json_durable

    def fail_after_manifest_replace(path: Path, payload, **kwargs) -> None:
        real_write_json_durable(path, payload, **kwargs)
        raise OSError("injected failure after manifest replacement")

    monkeypatch.setattr(
        promotion_module,
        "_write_json_durable",
        fail_after_manifest_replace,
    )

    with pytest.raises(OSError, match="after manifest replacement"):
        promote_cache_bundle(
            bundle.paths,
            destination,
            archive_root=tmp_path / "archives",
            expectations=bundle.expectations,
        )

    assert destination.document_artifact.read_bytes() == (
        bundle.paths.document_artifact.read_bytes()
    )
    assert destination.window_artifact.read_bytes() == (
        bundle.paths.window_artifact.read_bytes()
    )
    assert len(list((tmp_path / "archives").glob("*/promotion_manifest.json"))) == 1
    destination_cache = GlobalScoreCache(
        destination.score_cache_root,
        bundle.expectations.document_context,
    )
    assert destination_cache.lookup_many(
        [("query for topic 1", "document text for doc-1-1")]
    )[0] == 11.0
    assert destination_cache.connection.execute(
        "SELECT count(*) FROM imports"
    ).fetchone()[0] == 1
    destination_cache.close()


def test_failed_validation_leaves_live_files_untouched(tmp_path):
    bundle = _build_bundle(tmp_path)
    rows = _read_rows(bundle.paths.window_artifact)
    _write_jsonl(bundle.paths.window_artifact, rows[:-1])
    destination_document = tmp_path / "live" / "document.jsonl"
    destination_document.parent.mkdir(parents=True)
    destination_document.write_text("keep me\n", encoding="utf-8")
    destination = CacheBundlePaths(
        document_artifact=destination_document,
        window_artifact=tmp_path / "live" / "window.jsonl",
        score_cache_root=tmp_path / "live" / "score_cache",
    )
    archive_root = tmp_path / "archives"

    with pytest.raises(CacheBundleValidationError, match="incomplete window coverage"):
        promote_cache_bundle(
            bundle.paths,
            destination,
            archive_root=archive_root,
            expectations=bundle.expectations,
        )

    assert destination_document.read_text(encoding="utf-8") == "keep me\n"
    assert not archive_root.exists()


@pytest.mark.parametrize("legacy_policy", ["trec_rag_raw_v2", "extractive_sentence_pair_v1"])
def test_promotes_false_policy_bundle_through_authenticated_logical_rebinding(
    tmp_path, legacy_policy
):
    bundle = _build_bundle(tmp_path)
    target_document_context = replace(
        bundle.expectations.document_context,
        input_policy="trec_rag_whitespace_v1",
    )
    target_window_context = replace(
        bundle.expectations.window_context,
        input_policy="trec_rag_whitespace_v1",
    )
    legacy_document_context = replace(
        bundle.expectations.document_context, input_policy=legacy_policy
    )
    legacy_window_context = replace(
        bundle.expectations.window_context, input_policy=legacy_policy
    )
    expectations = replace(
        bundle.expectations,
        document_context=target_document_context,
        window_context=target_window_context,
    )
    staged = replace(
        bundle.paths,
        document_score_cache=bundle.document_cache_path,
        window_score_cache=bundle.window_cache_path,
    )
    destination, _, _ = _live_destination(
        tmp_path,
        bundle,
        document_context=target_document_context,
        window_context=target_window_context,
    )

    # The artifact and source cache retain the declared historical policy;
    # promotion must authenticate the source and re-key into the target cache.
    for artifact_path, context in (
        (
            bundle.paths.document_artifact,
            legacy_document_context,
        ),
        (
            bundle.paths.window_artifact,
            legacy_window_context,
        ),
    ):
        rows = _read_rows(artifact_path)
        for row in rows:
            query_hash = row["query_sha256"]
            text_hash = row["text_sha256"]
            if "document_text_sha256" in row:
                text_hash = row["text_sha256"]
            row.update(context.artifact_metadata)
            row["score_cache_key"] = promotion_module._cache_key_from_hashes(
                context, query_sha256=query_hash, text_sha256=text_hash
            )
        _write_jsonl(artifact_path, rows)
    _mutate_score_cache(
        bundle.document_cache_path,
        "UPDATE cache_meta SET value = ? WHERE key = 'context_json'",
        (legacy_document_context.context_json,),
    )
    _mutate_score_cache(
        bundle.document_cache_path,
        "UPDATE cache_meta SET value = ? WHERE key = 'context_sha256'",
        (legacy_document_context.context_sha256,),
    )
    _mutate_score_cache(
        bundle.window_cache_path,
        "UPDATE cache_meta SET value = ? WHERE key = 'context_json'",
        (legacy_window_context.context_json,),
    )
    _mutate_score_cache(
        bundle.window_cache_path,
        "UPDATE cache_meta SET value = ? WHERE key = 'context_sha256'",
        (legacy_window_context.context_sha256,),
    )
    # Rewrite source keys to the legacy context-bound formula.
    for path, legacy_context in (
        (bundle.document_cache_path, legacy_document_context),
        (bundle.window_cache_path, legacy_window_context),
    ):
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                "SELECT key_sha256, query_sha256, text_sha256 FROM scores"
            ).fetchall()
            for old_key, query_hash, text_hash in rows:
                legacy_key = promotion_module._cache_key_from_hashes(
                    legacy_context,
                    query_sha256=bytes(query_hash).hex(),
                    text_sha256=bytes(text_hash).hex(),
                )
                connection.execute(
                    "UPDATE scores SET key_sha256 = ? WHERE key_sha256 = ?",
                    (bytes.fromhex(legacy_key), old_key),
                )
                connection.commit()
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            for sidecar in (Path(f"{path}-wal"), Path(f"{path}-shm")):
                sidecar.unlink(missing_ok=True)

    result = promote_cache_bundle(
        staged,
        destination,
        archive_root=tmp_path / "archives",
        expectations=expectations,
    )
    manifest = json.loads(
        (result.archive_dir / "promotion_manifest.json").read_text(encoding="utf-8")
    )
    receipt = manifest["score_cache_receipts"]["document"]
    assert receipt["legacy_context_sha256"] == legacy_document_context.context_sha256
    assert receipt["target_context_sha256"] == target_document_context.context_sha256
    assert receipt["declared_input_policy"] == legacy_policy
    assert receipt["effective_input_policy"] == "trec_rag_whitespace_v1"
    assert receipt["source_sha256"] == result.validation.document_score_cache.sha256
    assert receipt["source_logical_digest"] == (
        result.validation.document_score_cache.logical_digest
    )
    assert len(receipt["target_logical_digest"]) == 64
    target_cache = GlobalScoreCache(destination.score_cache_root, target_document_context)
    assert target_cache.lookup_many([("query for topic 1", "document text for doc-1-1")])[0] == 11.0
    import_row = target_cache.connection.execute(
        "SELECT source_sha256, source_row_count, logical_digest, authorization_sha256 "
        "FROM imports"
    ).fetchone()
    assert import_row == (
        bytes.fromhex(receipt["source_sha256"]),
        receipt["source_row_count"],
        bytes.fromhex(receipt["target_logical_digest"]),
        bytes.fromhex(receipt["authorization_sha256"]),
    )
    target_cache.close()


def test_promotion_is_idempotent_and_safe_for_concurrent_identical_attempts(tmp_path):
    bundle = _build_bundle(tmp_path)
    destination, _, _ = _live_destination(tmp_path, bundle)
    archive_root = tmp_path / "archives"

    def promote_once():
        return promote_cache_bundle(
            bundle.paths,
            destination,
            archive_root=archive_root,
            expectations=bundle.expectations,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: promote_once(), range(2)))
    results.append(promote_once())

    target = GlobalScoreCache(
        destination.score_cache_root, bundle.expectations.document_context
    )
    assert target.lookup_many([("query for topic 2", "document text for doc-2-2")])[0] == 22.0
    target.close()
    assert len(list(archive_root.glob("*/promotion_manifest.json"))) == 3


def test_promotions_with_shared_destinations_serialize_across_archive_roots(
    tmp_path,
    monkeypatch,
):
    bundle = _build_bundle(tmp_path)
    destination, _, _ = _live_destination(tmp_path, bundle)
    first_inside = threading.Event()
    second_started = threading.Event()
    active_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_promote(*_args, **_kwargs):
        nonlocal active, maximum_active
        with active_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            is_first = not first_inside.is_set()
            if is_first:
                first_inside.set()
        if is_first:
            assert second_started.wait(timeout=1)
            time.sleep(0.05)
        with active_lock:
            active -= 1
        return object()

    monkeypatch.setattr(
        promotion_module,
        "_promote_cache_bundle_unlocked",
        fake_promote,
    )

    def promote_once(index: int):
        if index == 1:
            second_started.set()
        return promote_cache_bundle(
            bundle.paths,
            destination,
            archive_root=tmp_path / f"archives-{index}",
            expectations=bundle.expectations,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(promote_once, 0)
        assert first_inside.wait(timeout=1)
        second = executor.submit(promote_once, 1)
        first.result()
        second.result()

    assert maximum_active == 1


def test_promotion_rejects_conflicting_existing_destination_score_without_archiving(
    tmp_path,
):
    bundle = _build_bundle(tmp_path)
    destination, _, _ = _live_destination(tmp_path, bundle)
    target = GlobalScoreCache(
        destination.score_cache_root, bundle.expectations.document_context
    )
    target.add_many([("query for topic 1", "document text for doc-1-1", 999.0)])
    target.close()

    with pytest.raises(CacheBundleValidationError, match="destination|conflict"):
        promote_cache_bundle(
            bundle.paths,
            destination,
            archive_root=tmp_path / "archives",
            expectations=bundle.expectations,
        )
    assert not (tmp_path / "archives").exists()


def test_promotion_requires_contexts_derived_from_selected_config(tmp_path):
    bundle = _build_bundle(tmp_path)
    destination = CacheBundlePaths(
        document_artifact=tmp_path / "live" / "document.jsonl",
        window_artifact=tmp_path / "live" / "window.jsonl",
        score_cache_root=tmp_path / "live" / "score_cache",
    )
    expectations_without_context = replace(
        bundle.expectations,
        document_context=None,
        window_context=None,
    )

    with pytest.raises(
        CacheBundleValidationError,
        match="score contexts derived from the selected config",
    ):
        promote_cache_bundle(
            bundle.paths,
            destination,
            archive_root=tmp_path / "archives",
            expectations=expectations_without_context,
        )

    assert not (tmp_path / "archives").exists()
