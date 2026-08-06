from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import queue
import tarfile
from types import SimpleNamespace

import pytest
import zstandard

import trec_rag.competition_cache_bundle as bundle_module
from trec_rag.document_store import DocumentStore
from trec_rag.facet_pilot_config import (
    load_facet_pilot_config,
    select_configured_topics,
)
from trec_rag.planning_cache import PlanningCache, build_planning_cache_identity
from trec_rag.rerank_score_cache import GlobalScoreCache, ScoreCacheContext
from trec_rag.retrieval_cache import (
    DerivationIdentity,
    OrganizerTextNormalizer,
    RetrievalCache,
    TransportIdentity,
)
from trec_rag.similarity_cache import (
    SimilarityCache,
    build_similarity_cache_identity,
)
from trec_rag.topic_dispatch import (
    TopicJob,
    TopicJobReceipt,
    publish_topic_receipt,
)

from trec_rag.competition_cache_bundle import (
    CacheBundleConflictError,
    CacheBundleIntegrityError,
    MERGE_STATE_DIRECTORY,
    assert_no_incomplete_cache_bundle_merge,
    incomplete_merge_journals,
    merge_bundles,
    pack_bundle,
    verify_bundle,
)


_REAL_PRODUCTION_CHECKPOINT_VALIDATOR = (
    bundle_module._validate_production_topic_checkpoint
)

_CACHE_OPERATION_PHASES = ("planning", "retrieval", "scoring", "canonical")
_CACHE_OPERATION_STAGES = (
    "planning",
    "retrieval",
    "passage_scores",
    "sentence_scores",
    "similarity",
    "canonicalization",
)
_CACHE_OPERATION_COUNTERS = (
    "cache_hits",
    "cache_misses",
    "network_calls",
    "provider_calls",
    "model_batches",
)
_CACHE_OPERATION_TAMPER_CASES = (
    "malformed",
    "noncanonical",
    "forged_digest",
    "extra_field",
    "schema",
    "mode",
    "config",
    "topic",
    "run",
    "projection",
    "phase_names",
    "phase_value",
    "stage_names",
    "counter_names",
    "counter_bool",
    "counter_negative",
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _multiprocess_merge_worker(
    cache_root: str,
    outputs_root: str,
    bundle: str,
    start: object,
    results: object,
    pause_after_score_import: object | None = None,
    release_after_score_import: object | None = None,
) -> None:
    """Run one real process-level merge for the WAL publication regression."""
    try:
        if not start.wait(10):
            raise RuntimeError("multiprocess merge start barrier timed out")

        def publication_hook(phase: str) -> None:
            if not phase.startswith("score-imported:"):
                return
            if pause_after_score_import is None or release_after_score_import is None:
                return
            pause_after_score_import.set()
            if not release_after_score_import.wait(10):
                raise RuntimeError("score-import release barrier timed out")

        receipt = merge_bundles(
            cache_root=Path(cache_root),
            outputs_root=Path(outputs_root),
            bundle_dirs=(Path(bundle),),
            publication_hook=publication_hook,
        )
        results.put(("ok", receipt.merge_id))
    except BaseException as exc:  # pragma: no cover - asserted in the parent
        results.put(("error", f"{type(exc).__name__}: {exc}"))


def _online_cache_operation_receipt_bytes(
    *,
    config_sha256: str,
    projection_manifest_sha256: str,
) -> bytes:
    content: dict[str, object] = {
        "schema_version": "cache-operation-receipt-v1",
        "mode": "online",
        "run_id": "shard-fixture",
        "topic_id": "rag2026-0",
        "config_sha256": config_sha256,
        "projection_manifest_sha256": projection_manifest_sha256,
        "phases": {phase: {"resumed": False} for phase in _CACHE_OPERATION_PHASES},
        "stages": {
            stage: {counter: 0 for counter in _CACHE_OPERATION_COUNTERS}
            for stage in _CACHE_OPERATION_STAGES
        },
    }
    return _canonical_json(
        {
            **content,
            "receipt_content_sha256": hashlib.sha256(
                _canonical_json(content)
            ).hexdigest(),
        }
    )


@pytest.fixture(autouse=True)
def _use_synthetic_checkpoint_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep tiny bundle fixtures while production uses the deep validator."""
    monkeypatch.setattr(
        bundle_module,
        "_validate_production_topic_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bundle_module,
        "_validate_extracted_offline_replay",
        lambda *_args, **_kwargs: None,
    )


def _write_fixture(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text("fixture\n", encoding="utf-8")
    topics = root / "topics.tsv"
    topics.write_text("rag2026-0\tA tiny narrative.\n", encoding="utf-8")
    config = root / "config.yaml"
    config.write_text(
        """\
schema_version: facet_pilot_config_v2
experiment:
  id: shard-fixture
topics:
  path: topics.tsv
retrieval:
  index: climbmix-400b
  cache_dir: cache/retrieval/pyserini_remote
  query_sources: [original, subnarrative]
  documents_per_query: 1000
  corpus_epoch: fixture-epoch
passage:
  model: mixedbread-ai/mxbai-rerank-base-v2
  score_cache_dir: cache/reranker
  device: cpu
  passages_per_query: 100
  chunk_max_characters: 3500
  chunk_overlap_characters: 350
nuggets:
  evidence_budget_per_subnarrative: 40
  maximum_claims_per_subnarrative: 20
  maximum_supporting_documents_per_claim: 3
""",
        encoding="utf-8",
    )

    topic_root = root / "outputs" / "shard-fixture" / "rag2026-0"
    artifact_sets = {
        "retrieval": (
            "decomposition.json",
            "retrieval/audit.json",
            "retrieval/evidence-bundle.json",
        ),
        "scoring": (
            "scoring/lane_scores.jsonl",
            "scoring/selected_documents.jsonl",
            "scoring/selection.json",
            "scoring/selected_subnarrative_scores.jsonl",
        ),
        "canonical": (
            "canonical/handoff/candidate-requests.jsonl",
            "canonical/handoff/selection-contexts.jsonl",
            "canonical/handoff/handoff-manifest.json",
            "records.sqlite3",
            "canonical/records-manifest.json",
            "canonical/subnarrative-selections.jsonl",
            "canonical/selection-manifest.json",
            "canonical/canonical-nuggets.jsonl",
            "canonical/canonical-nugget-manifest.json",
            "canonical/retrieval-projection.json",
            "canonical/retrieval-projection-manifest.json",
            "canonical/generation-projection.json",
            "canonical/generation-projection-manifest.json",
        ),
    }
    for relative in {path for paths in artifact_sets.values() for path in paths}:
        path = topic_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "records.sqlite3":
            path.write_bytes(b"authenticated topic-records fixture")
        else:
            path.write_bytes(
                _canonical_json({"fixture": relative, "topic_id": "rag2026-0"})
            )
    decomposition_result = topic_root / "decomposition/result.json"
    decomposition_result.parent.mkdir(parents=True)
    decomposition_result.write_bytes(
        _canonical_json({"fixture": "decomposition source", "topic_id": "rag2026-0"})
    )
    decomposition_body = decomposition_result.read_bytes()
    (decomposition_result.parent / "manifest.json").write_bytes(
        _pretty_json(
            {
                "planner": {"fixture": True},
                "result_bytes": len(decomposition_body),
                "result_file": "result.json",
                "result_sha256": hashlib.sha256(decomposition_body).hexdigest(),
                "schema_version": "facet-decomposition-manifest-v1",
            }
        )
    )
    for phase, relatives in artifact_sets.items():
        manifest = topic_root / phase / "complete.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_bytes(
            _pretty_json(
                {
                    "artifacts": [
                        {
                            "bytes": (topic_root / relative).stat().st_size,
                            "relative_path": relative,
                            "sha256": hashlib.sha256(
                                (topic_root / relative).read_bytes()
                            ).hexdigest(),
                        }
                        for relative in relatives
                    ],
                    **(
                        {
                            "decomposition_source_sha256": hashlib.sha256(
                                decomposition_body
                            ).hexdigest(),
                            "retriever": {"fixture": True},
                        }
                        if phase == "retrieval"
                        else {}
                    ),
                    "phase": {
                        "retrieval": "retrieve",
                        "scoring": "score",
                        "canonical": "canonical",
                    }[phase],
                    "topic_id": "rag2026-0",
                }
            )
        )
    config_bytes = config.read_bytes()
    job = TopicJob(
        topic_id="rag2026-0",
        run_id="shard-fixture",
        config_path=config.resolve(),
        config_bytes=config_bytes,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        topic_root=topic_root.resolve(),
    )
    projection = topic_root / "canonical/retrieval-projection-manifest.json"
    (topic_root / "cache-operation-receipt.json").write_bytes(
        _online_cache_operation_receipt_bytes(
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            projection_manifest_sha256=hashlib.sha256(
                projection.read_bytes()
            ).hexdigest(),
        )
    )
    publish_topic_receipt(
        job,
        TopicJobReceipt(
            topic_id="rag2026-0",
            projection_manifest_sha256=hashlib.sha256(
                projection.read_bytes()
            ).hexdigest(),
            status="complete",
            stopping_reason="coverage_sufficient",
        ),
    )
    for export_name in (
        "cache-operation-manifest.json",
        "generation_handoff_manifest.json",
        "r_output_trec_rag_2026.tsv",
        "retrieval_export_manifest.json",
        "retrieval_with_text.jsonl.zip",
    ):
        (topic_root.parent / export_name).write_bytes(
            _canonical_json({"fixture": export_name})
        )

    identity = build_planning_cache_identity(
        request_body=b'{"messages":[]}',
        endpoint="https://openrouter.example/api",
        model="deepseek/fixture",
        prompt_version="fixture-v1",
        schema_version="fixture-schema-v1",
        topic_id="rag2026-0",
        narrative="A tiny narrative.",
    )
    PlanningCache(root / "cache").store(identity, {"validated": True})
    return config


def _planning_entry(root: Path) -> Path:
    return next((root / "planning-cache-v1").glob("*/*.json"))


def _write_manual_bundle(
    root: Path,
    *,
    declarations: list[dict[str, object]],
    entries: list[tuple[tarfile.TarInfo, bytes]],
    trailing: bytes = b"",
    source_config_sha256: str = "a" * 64,
) -> Path:
    root.mkdir()
    manifest = _canonical_json(
        {
            "experiment_id": "fixture",
            "members": declarations,
            "schema_version": "trec-rag-cache-bundle-v1",
            "source_config_sha256": source_config_sha256,
            "topic_id": "rag2026-0",
        }
    )
    tar_buffer = io.BytesIO()
    with tarfile.open(
        fileobj=tar_buffer, mode="w", format=tarfile.USTAR_FORMAT
    ) as archive:
        manifest_info = _tar_entry("bundle-manifest.json", manifest)
        archive.addfile(manifest_info, io.BytesIO(manifest))
        for info, body in entries:
            archive.addfile(info, io.BytesIO(body) if info.isfile() else None)
    compressed = zstandard.ZstdCompressor(
        level=3, write_checksum=True, write_content_size=False
    ).compress(tar_buffer.getvalue() + trailing)
    (root / "bundle.tar.zst").write_bytes(compressed)
    (root / "bundle-complete.json").write_bytes(
        _canonical_json(
            {
                "archive": "bundle.tar.zst",
                "archive_sha256": hashlib.sha256(compressed).hexdigest(),
                "archive_size": len(compressed),
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "member_count": len(declarations),
                "schema_version": "trec-rag-cache-bundle-v1",
                "topic_id": "rag2026-0",
            }
        )
    )
    return root


def _rewrite_bundle_member(
    bundle: Path,
    *,
    path: str,
    body: bytes | None,
    kind: str = "checkpoint",
) -> None:
    archive_path = bundle / "bundle.tar.zst"
    with zstandard.ZstdDecompressor().stream_reader(
        io.BytesIO(archive_path.read_bytes())
    ) as source:
        tar_bytes = source.read()
    bodies: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
        members = archive.getmembers()
        manifest_source = archive.extractfile(members[0])
        assert manifest_source is not None
        manifest = json.loads(manifest_source.read())
        for member in members[1:]:
            source = archive.extractfile(member)
            assert source is not None
            bodies[member.name] = source.read()
    declarations = {row["path"]: row for row in manifest["members"]}
    if body is None:
        bodies.pop(path, None)
        declarations.pop(path, None)
    else:
        bodies[path] = body
        declarations[path] = _declaration(path, body, kind=kind)
    manifest["members"] = [declarations[name] for name in sorted(declarations)]
    manifest_body = _canonical_json(manifest)
    rebuilt = io.BytesIO()
    with tarfile.open(
        fileobj=rebuilt, mode="w", format=tarfile.USTAR_FORMAT
    ) as archive:
        archive.addfile(
            _tar_entry("bundle-manifest.json", manifest_body),
            io.BytesIO(manifest_body),
        )
        for name in sorted(bodies):
            member_body = bodies[name]
            archive.addfile(_tar_entry(name, member_body), io.BytesIO(member_body))
    compressed = zstandard.ZstdCompressor(
        level=3,
        write_checksum=True,
        write_content_size=False,
    ).compress(rebuilt.getvalue())
    archive_path.write_bytes(compressed)
    (bundle / "bundle-complete.json").write_bytes(
        _canonical_json(
            {
                "archive": "bundle.tar.zst",
                "archive_sha256": hashlib.sha256(compressed).hexdigest(),
                "archive_size": len(compressed),
                "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
                "member_count": len(manifest["members"]),
                "schema_version": manifest["schema_version"],
                "topic_id": manifest["topic_id"],
            }
        )
    )


def _tampered_cache_operation_receipt(body: bytes, case: str) -> bytes:
    if case == "malformed":
        return b'{"broken":\n'
    value = json.loads(body)
    if case == "noncanonical":
        return _pretty_json(value)
    if case == "forged_digest":
        value["receipt_content_sha256"] = "0" * 64
        return _canonical_json(value)
    if case == "extra_field":
        value["unexpected"] = True
    elif case == "schema":
        value["schema_version"] = "cache-operation-receipt-v2"
    elif case == "mode":
        value["mode"] = "offline-cache-only"
    elif case == "config":
        value["config_sha256"] = "e" * 64
    elif case == "topic":
        value["topic_id"] = "rag2026-1"
    elif case == "run":
        value["run_id"] = "another-run"
    elif case == "projection":
        value["projection_manifest_sha256"] = "f" * 64
    elif case == "phase_names":
        del value["phases"]["canonical"]
    elif case == "phase_value":
        value["phases"]["canonical"]["resumed"] = 1
    elif case == "stage_names":
        del value["stages"]["similarity"]
    elif case == "counter_names":
        del value["stages"]["retrieval"]["network_calls"]
    elif case == "counter_bool":
        value["stages"]["retrieval"]["cache_hits"] = True
    elif case == "counter_negative":
        value["stages"]["retrieval"]["cache_misses"] = -1
    else:
        raise AssertionError(f"unknown receipt tamper case: {case}")
    content = dict(value)
    content.pop("receipt_content_sha256")
    value["receipt_content_sha256"] = hashlib.sha256(
        _canonical_json(content)
    ).hexdigest()
    return _canonical_json(value)


def _tar_entry(name: str, body: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(body)
    info.mode = 0o600
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _declaration(
    path: str, body: bytes, *, kind: str = "immutable"
) -> dict[str, object]:
    return {
        "kind": kind,
        "mode": 0o600,
        "path": path,
        "sha256": hashlib.sha256(body).hexdigest(),
        "size": len(body),
    }


def _manual_config_bytes() -> bytes:
    return b"""\
schema_version: facet_pilot_config_v2
experiment:
  id: fixture
topics:
  path: topics.tsv
retrieval:
  index: climbmix-400b
  cache_dir: cache/retrieval/pyserini_remote
  query_sources: [original, subnarrative]
  documents_per_query: 1000
  corpus_epoch: fixture-epoch
passage:
  model: mixedbread-ai/mxbai-rerank-base-v2
  score_cache_dir: cache/reranker
  device: cpu
  passages_per_query: 100
  chunk_max_characters: 3500
  chunk_overlap_characters: 350
nuggets:
  evidence_budget_per_subnarrative: 40
  maximum_claims_per_subnarrative: 20
  maximum_supporting_documents_per_claim: 3
"""


def test_pack_is_byte_reproducible_and_verify_is_manifest_driven(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_receipt = pack_bundle(config, "rag2026-0", first.resolve())
    second_receipt = pack_bundle(config, "rag2026-0", second.resolve())

    assert (first / "bundle.tar.zst").read_bytes() == (
        second / "bundle.tar.zst"
    ).read_bytes()
    assert (first / "bundle-complete.json").read_bytes() == (
        second / "bundle-complete.json"
    ).read_bytes()
    assert first_receipt == second_receipt
    verified = verify_bundle(first)
    assert verified.topic_id == "rag2026-0"
    assert [member.path for member in verified.members] == sorted(
        member.path for member in verified.members
    )
    assert any(
        member.path.startswith("cache/planning-cache-v1/")
        for member in verified.members
    )
    assert "outputs/shard-fixture/rag2026-0/retrieval/complete.json" in {
        member.path for member in verified.members
    }


def test_pack_publishes_nothing_when_offline_replay_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    destination = (tmp_path / "bundle").resolve()

    def reject_replay(*_args, **_kwargs):
        raise CacheBundleIntegrityError("offline replay fixture rejected")

    monkeypatch.setattr(
        bundle_module,
        "_validate_extracted_offline_replay",
        reject_replay,
    )

    with pytest.raises(CacheBundleIntegrityError, match="offline replay"):
        pack_bundle(config, "rag2026-0", destination)

    assert tuple(destination.iterdir()) == ()


def test_pack_includes_online_operation_receipt_but_not_root_exports(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()

    pack_bundle(config, "rag2026-0", bundle)
    members = {member.path: member for member in verify_bundle(bundle).members}

    operation_path = "outputs/shard-fixture/rag2026-0/cache-operation-receipt.json"
    assert members[operation_path].kind == "checkpoint"
    assert set(members).isdisjoint(
        {
            "outputs/shard-fixture/cache-operation-manifest.json",
            "outputs/shard-fixture/generation_handoff_manifest.json",
            "outputs/shard-fixture/r_output_trec_rag_2026.tsv",
            "outputs/shard-fixture/retrieval_export_manifest.json",
            "outputs/shard-fixture/retrieval_with_text.jsonl.zip",
        }
    )


def test_pack_requires_online_cache_operation_receipt(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    operation_path = (
        config.parent / "outputs/shard-fixture/rag2026-0/cache-operation-receipt.json"
    )
    operation_path.unlink()

    with pytest.raises(
        CacheBundleIntegrityError,
        match="cache operation receipt.*missing|missing.*cache operation receipt",
    ):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


@pytest.mark.parametrize(
    "case",
    _CACHE_OPERATION_TAMPER_CASES,
)
def test_pack_rejects_invalid_online_cache_operation_receipt(
    tmp_path: Path,
    case: str,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    operation_path = (
        config.parent / "outputs/shard-fixture/rag2026-0/cache-operation-receipt.json"
    )
    operation_path.write_bytes(
        _tampered_cache_operation_receipt(operation_path.read_bytes(), case)
    )

    with pytest.raises(CacheBundleIntegrityError, match="cache operation receipt"):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


@pytest.mark.parametrize(
    "case",
    _CACHE_OPERATION_TAMPER_CASES,
)
def test_verify_rejects_invalid_online_cache_operation_receipt_semantics(
    tmp_path: Path,
    case: str,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    operation_source = (
        config.parent / "outputs/shard-fixture/rag2026-0/cache-operation-receipt.json"
    )
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    _rewrite_bundle_member(
        bundle,
        path="outputs/shard-fixture/rag2026-0/cache-operation-receipt.json",
        body=_tampered_cache_operation_receipt(operation_source.read_bytes(), case),
    )

    with pytest.raises(CacheBundleIntegrityError, match="cache operation receipt"):
        verify_bundle(bundle)


def test_verify_requires_cache_operation_receipt_in_checkpoint_closure(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    _rewrite_bundle_member(
        bundle,
        path="outputs/shard-fixture/rag2026-0/cache-operation-receipt.json",
        body=None,
    )

    with pytest.raises(
        CacheBundleIntegrityError,
        match="cache operation receipt.*missing|missing.*cache operation receipt",
    ):
        verify_bundle(bundle)


def test_pack_requires_complete_document_cas_closure(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    missing_digest = hashlib.sha256(b"missing document").hexdigest()
    derivation = (
        config.parent
        / "cache/retrieval/pyserini_remote/v2/aa/request/derived/derivation"
    )
    derivation.mkdir(parents=True)
    (derivation / "derivation-manifest.json").write_bytes(
        _canonical_json({"document_closure": [missing_digest]})
    )

    with pytest.raises(CacheBundleIntegrityError, match="document.*closure"):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


def test_pack_validates_retrieval_derivation_and_exact_document_closure(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    query = "fixture query"
    identity = TransportIdentity.from_query(
        query_text=query,
        index_id="climbmix-400b",
        endpoint_identity="https://pyserini.example/search",
        corpus_epoch="fixture-epoch",
        hits=1000,
    )
    normalizer = OrganizerTextNormalizer()
    derivation = DerivationIdentity.from_normalizer(normalizer)
    retrieval = RetrievalCache(
        config.parent / "cache/retrieval/pyserini_remote",
        DocumentStore(config.parent / "cache/documents/v1"),
        normalizer,
    )
    body = "Exact organizer document body."
    raw = _canonical_json(
        {
            "api": "v1",
            "candidates": [{"doc": body, "docid": "doc-1", "rank": 1, "score": 3.5}],
            "index": "climbmix-400b",
            "query": {"text": query},
        }
    )[:-1]
    retrieval.commit(identity, derivation, query, raw)

    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    paths = {member.path for member in verify_bundle(bundle).members}
    document_digest = hashlib.sha256(body.encode()).hexdigest()

    assert (
        f"cache/documents/v1/sha256/{document_digest[:2]}/{document_digest}.utf8"
        in paths
    )
    assert any(path.endswith("/transport-manifest.json") for path in paths)
    assert any(path.endswith("/derivation-manifest.json") for path in paths)


def test_pack_requires_validated_planning_entries_for_selected_topic(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    entry = _planning_entry(config.parent / "cache")
    first = (tmp_path / "first").resolve()

    pack_bundle(config, "rag2026-0", first)

    assert f"cache/{entry.relative_to(config.parent / 'cache').as_posix()}" in {
        member.path for member in verify_bundle(first).members
    }
    entry.write_bytes(entry.read_bytes().removesuffix(b"\n") + b" \n")
    with pytest.raises(CacheBundleIntegrityError, match="planning cache"):
        pack_bundle(config, "rag2026-0", (tmp_path / "tampered").resolve())


def test_pack_rejects_fake_empty_checkpoint_seal(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    seal = config.parent / "outputs/shard-fixture/rag2026-0/retrieval/complete.json"
    seal.write_bytes(_canonical_json({}))

    with pytest.raises(
        CacheBundleIntegrityError, match="checkpoint|dispatch|projection"
    ):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


def test_pack_requires_production_topic_checkpoint_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_fixture(tmp_path / "repo")

    def reject(*_args: object, **_kwargs: object) -> None:
        raise CacheBundleIntegrityError("production checkpoint rejected")

    monkeypatch.setattr(bundle_module, "_validate_production_topic_checkpoint", reject)

    with pytest.raises(CacheBundleIntegrityError, match="production checkpoint"):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


def test_production_checkpoint_validator_cross_binds_dispatch_and_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_fixture(tmp_path / "repo")
    config_bytes = config_path.read_bytes()
    config = load_facet_pilot_config(config_path, source_bytes=config_bytes)
    topic = select_configured_topics(config, topic_ids=("rag2026-0",))[0]
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    producer_sha256 = "d" * 64
    projection_sha256 = hashlib.sha256(
        (
            config.output_dir
            / topic.id
            / "canonical/retrieval-projection-manifest.json"
        ).read_bytes()
    ).hexdigest()
    projection = SimpleNamespace(
        source_seals=(
            ("config_sha256", config_sha256),
            ("decomposition_producer_sha256", producer_sha256),
        ),
        manifest_sha256=projection_sha256,
        retrieval_status="complete",
        retrieval_stopping_reason="coverage_sufficient",
    )
    dispatch = SimpleNamespace(
        topic_id=topic.id,
        projection_manifest_sha256=projection_sha256,
        status="complete",
        stopping_reason="coverage_sufficient",
    )
    captured: dict[str, object] = {}

    import trec_rag.competition_retrieval as retrieval_runner
    import trec_rag.retrieval_export as retrieval_export
    import trec_rag.topic_dispatch as topic_dispatch

    monkeypatch.setattr(
        retrieval_runner,
        "_decomposition_producer_sha256",
        lambda *_args, **_kwargs: producer_sha256,
    )
    monkeypatch.setattr(
        retrieval_export,
        "read_topic_projection_receipt",
        lambda *_args, **_kwargs: projection,
    )

    def validate(*_args: object, **kwargs: object) -> tuple[object, ...]:
        captured.update(kwargs)
        return (projection,)

    monkeypatch.setattr(
        retrieval_export, "validate_retrieval_topic_checkpoints", validate
    )
    monkeypatch.setattr(
        topic_dispatch, "read_topic_receipt", lambda *_args, **_kwargs: dispatch
    )

    _REAL_PRODUCTION_CHECKPOINT_VALIDATOR(
        config,
        topic,
        config_path=config_path,
        config_bytes=config_bytes,
    )

    assert captured["expected_retriever_identity"] == {"fixture": True}
    assert captured["expected_decomposition_producer_sha256"] == {
        topic.id: producer_sha256
    }


def test_pack_rejects_lone_projection_manifest_as_a_topic_seal(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    topic_root = config.parent / "outputs/shard-fixture/rag2026-0"
    for path in sorted(topic_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    projection = topic_root / "canonical/retrieval-projection-manifest.json"
    projection.parent.mkdir(parents=True, exist_ok=True)
    projection.write_bytes(_canonical_json({"topic_id": "rag2026-0"}))

    with pytest.raises(
        CacheBundleIntegrityError, match="checkpoint|dispatch|projection"
    ):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


def test_pack_omits_unreceipted_topic_file_from_authenticated_closure(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    topic_root = config.parent / "outputs/shard-fixture/rag2026-0"
    extra = topic_root / "operator-note.json"
    extra.write_bytes(_canonical_json({"not": "checkpointed"}))
    bundle = (tmp_path / "bundle").resolve()

    pack_bundle(config, "rag2026-0", bundle)

    assert not any(
        member.path.endswith("operator-note.json")
        for member in verify_bundle(bundle).members
    )


def test_pack_rejects_stale_checkpoint_artifact_receipt(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    artifact = config.parent / "outputs/shard-fixture/rag2026-0/retrieval/audit.json"
    artifact.write_bytes(_canonical_json({"topic_id": "rag2026-X"}))

    with pytest.raises(CacheBundleIntegrityError, match="digest|hash|receipt|size"):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


@pytest.mark.parametrize(
    "relative",
    ("qrels/poison.json", "qrels.json", "provider-responses/raw.json"),
)
def test_pack_rejects_sensitive_unreceipted_topic_extra(
    tmp_path: Path,
    relative: str,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    poison = config.parent / "outputs/shard-fixture/rag2026-0" / relative
    poison.parent.mkdir(exist_ok=True)
    poison.write_bytes(_canonical_json({"private": True}))

    with pytest.raises(
        CacheBundleIntegrityError, match="forbidden|sensitive|prohibited"
    ):
        pack_bundle(config, "rag2026-0", (tmp_path / "bundle").resolve())


def test_verify_rejects_trailing_decompressed_payload(tmp_path: Path) -> None:
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[],
        entries=[],
        trailing=b"undeclared trailing payload",
    )

    with pytest.raises(CacheBundleIntegrityError, match="decompressed|trailing"):
        verify_bundle(bundle)


def test_verify_rejects_undeclared_bundle_directory_file(tmp_path: Path) -> None:
    bundle = _write_manual_bundle(tmp_path / "bundle", declarations=[], entries=[])
    (bundle / "provider-response.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(CacheBundleIntegrityError, match="undeclared"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../escape",
        "/absolute",
        "C:/windows-drive",
        "cache/../../escape",
        "cache\\..\\escape",
    ),
)
def test_verify_rejects_traversal_absolute_and_drive_paths(
    tmp_path: Path, unsafe_name: str
) -> None:
    body = b"unsafe"
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[_declaration(unsafe_name, body)],
        entries=[(_tar_entry(unsafe_name, body), body)],
    )

    with pytest.raises(CacheBundleIntegrityError, match="unsafe"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    "member_type",
    (
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.FIFOTYPE,
    ),
)
def test_verify_rejects_links_and_device_members(
    tmp_path: Path, member_type: bytes
) -> None:
    info = _tar_entry("cache/unsafe", b"")
    info.type = member_type
    info.linkname = "cache/target"
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[_declaration("cache/unsafe", b"")],
        entries=[(info, b"")],
    )

    with pytest.raises(CacheBundleIntegrityError, match="regular file"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    "paths",
    (
        ("cache/same", "cache/same"),
        ("cache/Entry.json", "cache/entry.json"),
        (
            "cache/caf\N{LATIN SMALL LETTER E WITH ACUTE}.json",
            "cache/cafe\N{COMBINING ACUTE ACCENT}.json",
        ),
    ),
)
def test_verify_rejects_duplicate_case_and_canonical_collisions(
    tmp_path: Path, paths: tuple[str, str]
) -> None:
    first, second = paths
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[_declaration(first, b"a"), _declaration(second, b"b")],
        entries=[(_tar_entry(first, b"a"), b"a"), (_tar_entry(second, b"b"), b"b")],
    )

    with pytest.raises(CacheBundleIntegrityError, match="duplicate|collision"):
        verify_bundle(bundle)


def test_verify_rejects_nonadjacent_ancestor_collision(tmp_path: Path) -> None:
    paths = ("a", "a-foo", "a/b")
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[_declaration(path, path.encode()) for path in paths],
        entries=[(_tar_entry(path, path.encode()), path.encode()) for path in paths],
    )

    with pytest.raises(CacheBundleIntegrityError, match="ancestor collision"):
        verify_bundle(bundle)


def test_verify_rejects_undeclared_archive_member(tmp_path: Path) -> None:
    body = b"undeclared"
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[],
        entries=[(_tar_entry("cache/extra", body), body)],
    )

    with pytest.raises(CacheBundleIntegrityError, match="undeclared"):
        verify_bundle(bundle)


@pytest.mark.parametrize("tamper", ("digest", "size"))
def test_verify_rejects_declared_digest_and_size_mismatch(
    tmp_path: Path, tamper: str
) -> None:
    body = b"authenticated"
    declaration = _declaration("cache/entry", body)
    if tamper == "digest":
        declaration["sha256"] = "0" * 64
    else:
        declaration["size"] = len(body) + 1
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[declaration],
        entries=[(_tar_entry("cache/entry", body), body)],
    )

    with pytest.raises(CacheBundleIntegrityError, match="digest|size"):
        verify_bundle(bundle)


def test_verify_enforces_member_and_decompression_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"two bytes"
    bundle = _write_manual_bundle(
        tmp_path / "member",
        declarations=[_declaration("cache/entry", body)],
        entries=[(_tar_entry("cache/entry", body), body)],
    )
    monkeypatch.setattr(bundle_module, "MAX_MEMBER_BYTES", 1)
    with pytest.raises(CacheBundleIntegrityError, match="size limit"):
        verify_bundle(bundle)

    empty = _write_manual_bundle(tmp_path / "decompressed", declarations=[], entries=[])
    monkeypatch.setattr(bundle_module, "MAX_MEMBER_BYTES", 1024)
    monkeypatch.setattr(bundle_module, "MAX_DECOMPRESSED_BYTES", 1024)
    with pytest.raises(CacheBundleIntegrityError, match="decompressed-size"):
        verify_bundle(empty)


def test_verify_rejects_compressed_portable_score_above_parser_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"x" * 4096
    path = f"portable-scores/{'a' * 64}.jsonl"
    bundle = _write_manual_bundle(
        tmp_path / "portable-score",
        declarations=[_declaration(path, body, kind="score")],
        entries=[(_tar_entry(path, body), body)],
    )
    monkeypatch.setattr(bundle_module, "MAX_PORTABLE_SCORE_BYTES", 1024)

    with pytest.raises(
        CacheBundleIntegrityError,
        match="portable score member size limit",
    ):
        verify_bundle(bundle)


def test_verify_rejects_prohibited_cache_member_even_when_self_declared(
    tmp_path: Path,
) -> None:
    config = _manual_config_bytes()
    poison = _canonical_json({"qrels": "private"})
    declarations = [
        _declaration("cache/qrels/poison.json", poison),
        _declaration("source-config/config.yaml", config, kind="config"),
    ]
    entries = [
        (_tar_entry("cache/qrels/poison.json", poison), poison),
        (_tar_entry("source-config/config.yaml", config), config),
    ]
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=declarations,
        entries=entries,
        source_config_sha256=hashlib.sha256(config).hexdigest(),
    )

    with pytest.raises(CacheBundleIntegrityError, match="forbidden|prohibited"):
        verify_bundle(bundle)


def test_verify_requires_exactly_one_authenticated_source_config(
    tmp_path: Path,
) -> None:
    missing = _write_manual_bundle(tmp_path / "missing", declarations=[], entries=[])
    with pytest.raises(CacheBundleIntegrityError, match="exactly one.*config"):
        verify_bundle(missing)

    config = _manual_config_bytes()
    duplicate_path = "source-config/duplicate.yaml"
    declarations = [
        _declaration("source-config/config.yaml", config, kind="config"),
        _declaration(duplicate_path, config, kind="config"),
    ]
    duplicate = _write_manual_bundle(
        tmp_path / "duplicate",
        declarations=declarations,
        entries=[
            (_tar_entry("source-config/config.yaml", config), config),
            (_tar_entry(duplicate_path, config), config),
        ],
        source_config_sha256=hashlib.sha256(config).hexdigest(),
    )
    with pytest.raises(CacheBundleIntegrityError, match="exactly one.*config"):
        verify_bundle(duplicate)


def test_verify_binds_source_config_digest_and_experiment(tmp_path: Path) -> None:
    config = _manual_config_bytes()
    declaration = _declaration("source-config/config.yaml", config, kind="config")
    entry = (_tar_entry("source-config/config.yaml", config), config)
    wrong_digest = _write_manual_bundle(
        tmp_path / "wrong-digest",
        declarations=[declaration],
        entries=[entry],
        source_config_sha256="0" * 64,
    )
    with pytest.raises(CacheBundleIntegrityError, match="config digest"):
        verify_bundle(wrong_digest)

    wrong_experiment_bytes = config.replace(b"id: fixture", b"id: other")
    wrong_experiment = _write_manual_bundle(
        tmp_path / "wrong-experiment",
        declarations=[
            _declaration(
                "source-config/config.yaml",
                wrong_experiment_bytes,
                kind="config",
            )
        ],
        entries=[
            (
                _tar_entry("source-config/config.yaml", wrong_experiment_bytes),
                wrong_experiment_bytes,
            )
        ],
        source_config_sha256=hashlib.sha256(wrong_experiment_bytes).hexdigest(),
    )
    with pytest.raises(CacheBundleIntegrityError, match="experiment"):
        verify_bundle(wrong_experiment)


@pytest.mark.parametrize(
    ("path", "kind"),
    (
        ("cache/planning-cache-v1/aa/" + "a" * 64 + ".json", "config"),
        ("source-config/config.yaml", "immutable"),
        ("cache/planning-cache-v1/aa/" + "a" * 64 + ".json", "checkpoint"),
        ("outputs/fixture/rag2026-0/retrieval/audit.json", "similarity"),
    ),
)
def test_verify_rejects_wrong_kind_path_combinations(
    tmp_path: Path,
    path: str,
    kind: str,
) -> None:
    config = _manual_config_bytes()
    body = config if path == "source-config/config.yaml" else _canonical_json({})
    declarations = [_declaration(path, body, kind=kind)]
    entries = [(_tar_entry(path, body), body)]
    if path != "source-config/config.yaml":
        declarations.append(
            _declaration("source-config/config.yaml", config, kind="config")
        )
        entries.append((_tar_entry("source-config/config.yaml", config), config))
    paired = sorted(
        zip(declarations, entries, strict=True), key=lambda pair: pair[0]["path"]
    )
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=[pair[0] for pair in paired],
        entries=[pair[1] for pair in paired],
        source_config_sha256=hashlib.sha256(config).hexdigest(),
    )

    with pytest.raises(CacheBundleIntegrityError, match="kind|config|path"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("path", "kind", "body"),
    (
        (
            "cache/planning-cache-v1/aa/" + "a" * 64 + ".json",
            "immutable",
            b'{"identity":{}}\n',
        ),
        (
            "cache/similarity-cache-v1/aa/" + "a" * 64 + ".json",
            "similarity",
            b'{"identity":{},"matrix_hex":[["nan"]]}\n',
        ),
    ),
)
def test_verify_rejects_malformed_cache_identity(
    tmp_path: Path,
    path: str,
    kind: str,
    body: bytes,
) -> None:
    config = _manual_config_bytes()
    declarations = [
        _declaration(path, body, kind=kind),
        _declaration("source-config/config.yaml", config, kind="config"),
    ]
    entries = [
        (_tar_entry(path, body), body),
        (_tar_entry("source-config/config.yaml", config), config),
    ]
    bundle = _write_manual_bundle(
        tmp_path / "bundle",
        declarations=declarations,
        entries=entries,
        source_config_sha256=hashlib.sha256(config).hexdigest(),
    )

    with pytest.raises(CacheBundleIntegrityError, match="planning|similarity|identity"):
        verify_bundle(bundle)


def test_merge_installs_verified_files_and_is_idempotent(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()

    first = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts="strict",
    )
    second = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts="strict",
    )

    assert second == first
    source_planning = _planning_entry(config.parent / "cache")
    assert _planning_entry(cache_root).read_bytes() == source_planning.read_bytes()
    assert (outputs_root / "shard-fixture/rag2026-0/retrieval/complete.json").is_file()
    state = cache_root / MERGE_STATE_DIRECTORY / first.merge_id
    assert (state / "prepare.json").is_file()
    assert (state / "conflicts.json").is_file()
    assert first.completion_path == state / "complete.json"
    assert first.completion_path.is_file()


def test_merge_preflights_strict_conflicts_before_destination_mutation(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    source_planning = _planning_entry(config.parent / "cache")
    conflicting = cache_root / source_planning.relative_to(config.parent / "cache")
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"destination authority")

    with pytest.raises(CacheBundleConflictError, match="immutable"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="strict",
        )

    assert conflicting.read_bytes() == b"destination authority"
    assert not outputs_root.exists()
    assert not (cache_root / MERGE_STATE_DIRECTORY).exists()


def test_merge_retry_converges_from_durable_prepare_journal(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    installed_events = 0

    def interrupt_after_first_install(phase: str) -> None:
        nonlocal installed_events
        if phase.startswith("installed:"):
            installed_events += 1
            if installed_events == 1:
                raise RuntimeError("simulated process interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            publication_hook=interrupt_after_first_install,
        )
    with pytest.raises(CacheBundleIntegrityError, match="incomplete"):
        assert_no_incomplete_cache_bundle_merge(cache_root)

    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )

    assert receipt.completion_path.is_file()
    assert receipt.operation_count == receipt.installed_count + receipt.identical_count
    assert_no_incomplete_cache_bundle_merge(cache_root)


def test_identical_merge_retry_adopts_authenticated_completion(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    first = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    completion_bytes = first.completion_path.read_bytes()
    retried = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )

    assert first == retried
    assert first.completion_path.read_bytes() == completion_bytes
    assert_no_incomplete_cache_bundle_merge(cache_root)


def test_merge_verifies_bundles_before_acquiring_the_destination_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    events: list[str] = []
    original_verify = bundle_module.verify_bundle
    original_serialized = bundle_module._serialized_merge

    def traced_verify(path):
        events.append("verify")
        return original_verify(path)

    @contextmanager
    def traced_serialized(cache_root):
        events.append("lock")
        with original_serialized(cache_root):
            yield

    monkeypatch.setattr(bundle_module, "verify_bundle", traced_verify)
    monkeypatch.setattr(bundle_module, "_serialized_merge", traced_serialized)

    merge_bundles(
        cache_root=(tmp_path / "cache").resolve(),
        outputs_root=(tmp_path / "outputs").resolve(),
        bundle_dirs=(bundle,),
    )

    assert events[:2] == ["verify", "lock"]


def test_merge_lock_rejects_same_thread_reentry_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = (tmp_path / "cache").resolve()
    monkeypatch.setattr(bundle_module.fcntl, "flock", lambda *_args: None)

    with bundle_module._serialized_merge(cache_root):
        with pytest.raises(CacheBundleIntegrityError, match="not reentrant"):
            with bundle_module._serialized_merge(cache_root):
                pytest.fail("same-thread nested merge acquired its own lock")


def test_invalid_bundle_creates_no_destination_or_merge_lock(
    tmp_path: Path,
) -> None:
    destination_parent = tmp_path / "destinations"

    with pytest.raises(CacheBundleIntegrityError):
        merge_bundles(
            cache_root=(destination_parent / "cache").resolve(),
            outputs_root=(destination_parent / "outputs").resolve(),
            bundle_dirs=(tmp_path / "missing-bundle",),
        )

    assert not destination_parent.exists()


def test_concurrent_score_merge_cannot_publish_while_first_merge_is_midflight(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    context = ScoreCacheContext(
        backend="fixture-backend",
        model="fixture-model",
        max_length=128,
        score_kind="passage",
        model_revision="fixture-revision",
        backend_version="1.0",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy="trec_rag_whitespace_v1",
    )
    source = GlobalScoreCache(config.parent / "cache/reranker", context)
    source.seed_many(
        [("query", "passage", 1.25)],
        source_path="multiprocess-source",
        source_sha256="1" * 64,
    )
    source.close()
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    process_context = multiprocessing.get_context("fork")
    first_start = process_context.Event()
    second_start = process_context.Event()
    paused = process_context.Event()
    release = process_context.Event()
    results = process_context.Queue()
    first = process_context.Process(
        target=_multiprocess_merge_worker,
        args=(
            str(cache_root),
            str(outputs_root),
            str(bundle),
            first_start,
            results,
            paused,
            release,
        ),
    )
    second = process_context.Process(
        target=_multiprocess_merge_worker,
        args=(
            str(cache_root),
            str(outputs_root),
            str(bundle),
            second_start,
            results,
        ),
    )
    processes = (first, second)
    first.start()
    first_start.set()
    assert paused.wait(10), "first merge never reached its score-import boundary"
    second.start()
    second_start.set()
    try:
        early = results.get(timeout=0.5)
    except queue.Empty:
        early = None
    finally:
        release.set()
    for process in processes:
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(5)
            pytest.fail("multiprocess merge worker timed out")
        assert process.exitcode == 0
    assert early is None, f"a competing merge published while midflight: {early}"
    outcomes = tuple(results.get(timeout=5) for _ in processes)
    assert {status for status, _payload in outcomes} == {"ok"}
    assert len({payload for _status, payload in outcomes}) == 1

    imported = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    try:
        assert imported.get(query_text="query", text="passage") == 1.25
        database = imported.path
    finally:
        imported.close()
    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()
    assert_no_incomplete_cache_bundle_merge(cache_root)


def test_concurrent_merge_rejects_unauthenticated_completion_winner(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    forged_paths = []
    forged_bodies = []

    def publish_forged_completion(phase: str) -> None:
        if phase != "prepare":
            return
        state_roots = tuple((cache_root / MERGE_STATE_DIRECTORY).iterdir())
        assert len(state_roots) == 1
        state_root = state_roots[0]
        prepare = (state_root / "prepare.json").read_bytes()
        forged = _canonical_json(
            {
                "bundle_count": 1,
                "conflicts_sha256": "0" * 64,
                "identical_count": 0,
                "installed_count": 0,
                "kept_conflict_count": 0,
                "merge_id": state_root.name,
                "operation_count": 0,
                "prepare_sha256": hashlib.sha256(prepare).hexdigest(),
                "schema_version": "trec-rag-cache-bundle-v1",
                "score_import_count": 0,
            }
        )
        complete_path = state_root / "complete.json"
        complete_path.write_bytes(forged)
        forged_paths.append(complete_path)
        forged_bodies.append(forged)

    with pytest.raises(CacheBundleConflictError, match="durable merge state"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            publication_hook=publish_forged_completion,
        )

    assert len(forged_paths) == 1
    assert forged_paths[0].read_bytes() == forged_bodies[0]


def test_pack_exports_portable_scores_and_merge_imports_transactionally(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    context = ScoreCacheContext(
        backend="fixture-backend",
        model="fixture-model",
        max_length=128,
        score_kind="passage",
        model_revision="fixture-revision",
        backend_version="1.0",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy="trec_rag_whitespace_v1",
    )
    source_scores = GlobalScoreCache(config.parent / "cache/reranker", context)
    source_scores.seed_many(
        [("query", "passage", 1.25)],
        source_path="fixture",
        source_sha256="1" * 64,
    )
    source_scores.close()
    bundle = (tmp_path / "bundle").resolve()

    pack_bundle(config, "rag2026-0", bundle)
    verified = verify_bundle(bundle)

    score_members = [member for member in verified.members if member.kind == "score"]
    assert [member.path for member in score_members] == [
        f"portable-scores/{context.context_sha256}.jsonl"
    ]
    assert not any(
        member.path.startswith("cache/") and member.path.endswith(".sqlite3")
        for member in verified.members
    )
    assert any(member.path.endswith("/records.sqlite3") for member in verified.members)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()

    def interrupt_after_score_import(phase: str) -> None:
        if phase.startswith("score-imported:"):
            raise RuntimeError("simulated post-transaction interruption")

    with pytest.raises(RuntimeError, match="post-transaction"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            publication_hook=interrupt_after_score_import,
        )
    with pytest.raises(CacheBundleIntegrityError, match="incomplete"):
        assert_no_incomplete_cache_bundle_merge(cache_root)
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    imported = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    try:
        assert imported.get(query_text="query", text="passage") == 1.25
    finally:
        imported.close()
    assert receipt.score_import_count == 1
    assert_no_incomplete_cache_bundle_merge(cache_root)


@pytest.mark.parametrize("score_conflicts", ("strict", "keep-existing"))
def test_completed_merge_reimports_safely_missing_portable_score_database(
    tmp_path: Path,
    score_conflicts: str,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    context = ScoreCacheContext(
        backend="fixture-backend",
        model="fixture-model",
        max_length=128,
        score_kind="passage",
        model_revision="fixture-revision",
        backend_version="1.0",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy="trec_rag_whitespace_v1",
    )
    source = GlobalScoreCache(config.parent / "cache/reranker", context)
    source.seed_many(
        [("query", "passage", 1.25)],
        source_path="source",
        source_sha256="1" * 64,
    )
    source.close()
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    first = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts=score_conflicts,
    )
    imported = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    database = imported.path
    imported.close()
    database.unlink()

    second = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts=score_conflicts,
    )

    restored = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    try:
        assert restored.get(query_text="query", text="passage") == 1.25
    finally:
        restored.close()
    assert second == first


def test_completed_merge_rejects_score_conflict_created_before_reimport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    context = ScoreCacheContext(
        backend="fixture-backend",
        model="fixture-model",
        max_length=128,
        score_kind="passage",
        model_revision="fixture-revision",
        backend_version="1.0",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy="trec_rag_whitespace_v1",
    )
    source = GlobalScoreCache(config.parent / "cache/reranker", context)
    source.seed_many(
        [("query", "passage", 1.25)],
        source_path="source",
        source_sha256="1" * 64,
    )
    source.close()
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    first = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts="keep-existing",
    )
    completion_bytes = first.completion_path.read_bytes()
    imported = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    database = imported.path
    imported.close()
    database.unlink()
    real_import = bundle_module._import_score_operation
    conflict_inserted = False

    def insert_conflict_then_import(
        operation,
        *,
        cache_root: Path,
        conflict_policy: str,
    ):
        nonlocal conflict_inserted
        if not conflict_inserted:
            destination = GlobalScoreCache(cache_root / "reranker", context)
            destination.seed_many(
                [("query", "passage", 2.5)],
                source_path="concurrent-recovery-writer",
                source_sha256="2" * 64,
            )
            destination.close()
            conflict_inserted = True
        return real_import(
            operation,
            cache_root=cache_root,
            conflict_policy=conflict_policy,
        )

    monkeypatch.setattr(
        bundle_module,
        "_import_score_operation",
        insert_conflict_then_import,
    )

    with pytest.raises(CacheBundleConflictError, match="score|conflict|audit"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
        )

    retained = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    try:
        assert retained.get(query_text="query", text="passage") == 2.5
    finally:
        retained.close()
    assert first.completion_path.read_bytes() == completion_bytes


def test_completed_merge_fails_closed_on_corrupt_portable_score_database(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    context = ScoreCacheContext(
        backend="fixture-backend",
        model="fixture-model",
        max_length=128,
        score_kind="passage",
        model_revision="fixture-revision",
        backend_version="1.0",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy="trec_rag_whitespace_v1",
    )
    source = GlobalScoreCache(config.parent / "cache/reranker", context)
    source.seed_many(
        [("query", "passage", 1.25)],
        source_path="source",
        source_sha256="1" * 64,
    )
    source.close()
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    imported = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    database = imported.path
    imported.close()
    database.write_bytes(b"not a sqlite database")

    with pytest.raises(CacheBundleIntegrityError, match="score|database|corrupt"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
        )


def test_completed_merge_rejects_impossible_canonical_receipt_counts(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    forged = json.loads(receipt.completion_path.read_bytes())
    forged["installed_count"] = 999_999
    receipt.completion_path.write_bytes(_canonical_json(forged))
    prepare_path = receipt.completion_path.parent / "prepare.json"

    assert incomplete_merge_journals(cache_root) == (prepare_path,)
    with pytest.raises(CacheBundleIntegrityError, match="completion|receipt|count"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
        )


def test_completed_merge_rejects_its_flagged_current_journal(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    forged = json.loads(receipt.completion_path.read_bytes())
    forged["prepare_sha256"] = "0" * 64
    receipt.completion_path.write_bytes(_canonical_json(forged))
    prepare_path = receipt.completion_path.parent / "prepare.json"

    assert incomplete_merge_journals(cache_root) == (prepare_path,)
    with pytest.raises(CacheBundleIntegrityError, match="completion|receipt|journal"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
        )


def test_completed_merge_rejects_nonregular_receipt_before_recovery_mutation(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    outputs_root = (tmp_path / "merged-outputs").resolve()
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
    )
    outside = tmp_path / "forged-complete.json"
    outside.write_bytes(receipt.completion_path.read_bytes())
    receipt.completion_path.unlink()
    receipt.completion_path.symlink_to(outside)
    publication_events: list[str] = []

    with pytest.raises(CacheBundleIntegrityError, match="completion|receipt|regular"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            publication_hook=publication_events.append,
        )

    assert publication_events == []


def test_keep_existing_rejects_score_conflict_created_after_preflight(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    context = ScoreCacheContext(
        backend="fixture-backend",
        model="fixture-model",
        max_length=128,
        score_kind="passage",
        model_revision="fixture-revision",
        backend_version="1.0",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy="trec_rag_whitespace_v1",
    )
    source = GlobalScoreCache(config.parent / "cache/reranker", context)
    source.seed_many(
        [("query", "passage", 1.25)],
        source_path="source",
        source_sha256="1" * 64,
    )
    source.close()
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    outputs_root = (tmp_path / "destination-outputs").resolve()

    def publish_conflicting_score(phase: str) -> None:
        if phase != "prepare":
            return
        destination = GlobalScoreCache(cache_root / "reranker", context)
        destination.seed_many(
            [("query", "passage", 2.5)],
            source_path="concurrent-writer",
            source_sha256="2" * 64,
        )
        destination.close()

    with pytest.raises(CacheBundleConflictError, match="score|conflict|preflight"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
            publication_hook=publish_conflicting_score,
        )

    retained = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    try:
        assert retained.get(query_text="query", text="passage") == 2.5
    finally:
        retained.close()
    states = tuple((cache_root / MERGE_STATE_DIRECTORY).iterdir())
    assert len(states) == 1
    assert not (states[0] / "complete.json").exists()


def test_keep_existing_rejects_similarity_conflict_changed_after_preflight(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    model_identity = {
        "backend": "sentence-transformers",
        "embedding_representation": "normalized-float32",
        "local_files_only": True,
        "model": "fixture/minilm",
        "model_revision": "revision",
        "score_kind": "cosine-similarity",
    }
    identity = build_similarity_cache_identity(
        model_identity=model_identity,
        texts=("first", "second"),
    )
    SimilarityCache(config.parent / "cache").store(
        identity,
        ((1.0, 0.25), (0.25, 1.0)),
    )
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    destination = SimilarityCache(cache_root).store(
        identity,
        ((1.0, 0.5), (0.5, 1.0)),
    )
    replacement = SimilarityCache(tmp_path / "replacement-cache").store(
        identity,
        ((1.0, 0.75), (0.75, 1.0)),
    )
    replacement_bytes = replacement.read_bytes()
    outputs_root = (tmp_path / "destination-outputs").resolve()

    def replace_similarity_conflict(phase: str) -> None:
        if phase == "prepare":
            destination.write_bytes(replacement_bytes)

    with pytest.raises(
        CacheBundleConflictError,
        match="similarity|conflict|preflight|destination",
    ):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
            publication_hook=replace_similarity_conflict,
        )

    assert destination.read_bytes() == replacement_bytes
    states = tuple((cache_root / MERGE_STATE_DIRECTORY).iterdir())
    assert len(states) == 1
    assert not (states[0] / "complete.json").exists()


def test_score_conflicts_are_strict_by_default_and_keep_existing_is_audited(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    context = ScoreCacheContext(
        backend="fixture-backend",
        model="fixture-model",
        max_length=128,
        score_kind="passage",
        model_revision="fixture-revision",
        backend_version="1.0",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy="trec_rag_whitespace_v1",
    )
    source = GlobalScoreCache(config.parent / "cache/reranker", context)
    source.seed_many(
        [("query", "passage", 1.25)],
        source_path="source",
        source_sha256="1" * 64,
    )
    source.close()
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)

    cache_root = (tmp_path / "destination-cache").resolve()
    destination = GlobalScoreCache(cache_root / "reranker", context)
    destination.seed_many(
        [("query", "passage", 2.5)],
        source_path="destination",
        source_sha256="2" * 64,
    )
    destination.close()
    outputs_root = (tmp_path / "destination-outputs").resolve()
    with pytest.raises(CacheBundleConflictError, match="numerical score"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
        )
    assert not outputs_root.exists()
    assert not (cache_root / MERGE_STATE_DIRECTORY).exists()

    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts="keep-existing",
    )
    retained = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    try:
        assert retained.get(query_text="query", text="passage") == 2.5
    finally:
        retained.close()
    audit = json.loads(
        (receipt.completion_path.parent / "conflicts.json").read_text(encoding="utf-8")
    )
    assert audit["conflicts"] == [
        {
            "cache_key": audit["conflicts"][0]["cache_key"],
            "context_sha256": context.context_sha256,
            "existing_score_hex": (2.5).hex(),
            "kind": "score",
            "query_sha256": hashlib.sha256(b"query").hexdigest(),
            "resolution": "kept-existing",
            "scope": "destination",
            "source_score_hex": (1.25).hex(),
            "text_sha256": hashlib.sha256(b"passage").hexdigest(),
        }
    ]


def test_multi_bundle_keep_existing_retry_preserves_canonical_bundle_conflict(
    tmp_path: Path,
) -> None:
    context = ScoreCacheContext(
        backend="fixture-backend",
        model="fixture-model",
        max_length=128,
        score_kind="passage",
        model_revision="fixture-revision",
        backend_version="1.0",
        score_representation="raw_logits",
        inference_dtype="float32",
        input_policy="trec_rag_whitespace_v1",
    )
    bundles = []
    ordered_inputs = []
    cache_key = ""
    for label, value in (("first", 1.25), ("second", 2.5)):
        config = _write_fixture(tmp_path / f"repo-{label}")
        source = GlobalScoreCache(config.parent / "cache/reranker", context)
        source.seed_many(
            [("query", "passage", value)],
            source_path=label,
            source_sha256=("1" if label == "first" else "2") * 64,
        )
        cache_key = source.cache_key(query_text="query", text="passage")
        source.close()
        bundle = (tmp_path / f"bundle-{label}").resolve()
        pack_bundle(config, "rag2026-0", bundle)
        score_member = next(
            member for member in verify_bundle(bundle).members if member.kind == "score"
        )
        bundles.append(bundle)
        ordered_inputs.append((score_member.sha256, value))
    winner, conflicting = (
        value for _digest, value in sorted(ordered_inputs, key=lambda item: item[0])
    )
    cache_root = (tmp_path / "destination-cache").resolve()
    outputs_root = (tmp_path / "destination-outputs").resolve()

    first = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=tuple(bundles),
        score_conflicts="keep-existing",
    )
    conflict_path = first.completion_path.parent / "conflicts.json"
    first_conflict_bytes = conflict_path.read_bytes()
    first_audit = json.loads(first_conflict_bytes)
    second = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=tuple(reversed(bundles)),
        score_conflicts="keep-existing",
    )

    retained = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
    try:
        assert retained.get(query_text="query", text="passage") == winner
    finally:
        retained.close()
    assert second == first
    assert conflict_path.read_bytes() == first_conflict_bytes
    assert first_audit["conflicts"] == [
        {
            "cache_key": cache_key,
            "context_sha256": context.context_sha256,
            "existing_score_hex": winner.hex(),
            "kind": "score",
            "query_sha256": hashlib.sha256(b"query").hexdigest(),
            "resolution": "kept-existing",
            "scope": "bundle",
            "source_score_hex": conflicting.hex(),
            "text_sha256": hashlib.sha256(b"passage").hexdigest(),
        }
    ]


def test_keep_existing_is_limited_to_similarity_and_audits_numerical_conflict(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    model_identity = {
        "backend": "sentence-transformers",
        "embedding_representation": "normalized-float32",
        "local_files_only": True,
        "model": "fixture/minilm",
        "model_revision": "revision",
        "score_kind": "cosine-similarity",
    }
    identity = build_similarity_cache_identity(
        model_identity=model_identity,
        texts=("first", "second"),
    )
    SimilarityCache(config.parent / "cache").store(identity, ((1.0, 0.25), (0.25, 1.0)))
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    SimilarityCache(cache_root).store(identity, ((1.0, 0.5), (0.5, 1.0)))

    with pytest.raises(CacheBundleConflictError, match="immutable"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=(tmp_path / "strict-outputs").resolve(),
            bundle_dirs=(bundle,),
        )
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=(tmp_path / "kept-outputs").resolve(),
        bundle_dirs=(bundle,),
        score_conflicts="keep-existing",
    )

    assert SimilarityCache(cache_root).load(identity) == (
        (1.0, 0.5),
        (0.5, 1.0),
    )
    audit = json.loads(
        (receipt.completion_path.parent / "conflicts.json").read_text(encoding="utf-8")
    )
    assert [(row["kind"], row["resolution"]) for row in audit["conflicts"]] == [
        ("similarity", "kept-existing")
    ]


def test_completed_keep_existing_merge_rejects_changed_similarity_target(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    model_identity = {
        "backend": "sentence-transformers",
        "embedding_representation": "normalized-float32",
        "local_files_only": True,
        "model": "fixture/minilm",
        "model_revision": "revision",
        "score_kind": "cosine-similarity",
    }
    identity = build_similarity_cache_identity(
        model_identity=model_identity,
        texts=("first", "second"),
    )
    source_cache = SimilarityCache(config.parent / "cache")
    source_cache.store(identity, ((1.0, 0.25), (0.25, 1.0)))
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    destination_cache = SimilarityCache(cache_root)
    destination = destination_cache.store(identity, ((1.0, 0.5), (0.5, 1.0)))
    outputs_root = (tmp_path / "destination-outputs").resolve()
    receipt = merge_bundles(
        cache_root=cache_root,
        outputs_root=outputs_root,
        bundle_dirs=(bundle,),
        score_conflicts="keep-existing",
    )
    conflict_path = receipt.completion_path.parent / "conflicts.json"
    original_conflict_bytes = conflict_path.read_bytes()
    replacement = SimilarityCache(tmp_path / "replacement-cache").store(
        identity,
        ((1.0, 0.75), (0.75, 1.0)),
    )
    destination.write_bytes(replacement.read_bytes())

    with pytest.raises(
        CacheBundleIntegrityError, match="completion|conflict|destination"
    ):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
        )

    assert conflict_path.read_bytes() == original_conflict_bytes


def test_keep_existing_rejects_malformed_destination_similarity_before_mutation(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    model_identity = {
        "backend": "sentence-transformers",
        "embedding_representation": "normalized-float32",
        "local_files_only": True,
        "model": "fixture/minilm",
        "model_revision": "revision",
        "score_kind": "cosine-similarity",
    }
    identity = build_similarity_cache_identity(
        model_identity=model_identity,
        texts=("first", "second"),
    )
    SimilarityCache(config.parent / "cache").store(identity, ((1.0, 0.25), (0.25, 1.0)))
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    destination = SimilarityCache(cache_root).entry_path(identity)
    destination.parent.mkdir(parents=True)
    malformed = json.loads(
        SimilarityCache(config.parent / "cache").entry_path(identity).read_bytes()
    )
    malformed["matrix_hex"][0][0] = "nan"
    malformed["matrix_sha256"] = hashlib.sha256(
        json.dumps(
            malformed["matrix_hex"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    destination.write_bytes(_canonical_json(malformed))
    outputs_root = (tmp_path / "destination-outputs").resolve()

    with pytest.raises(CacheBundleIntegrityError, match="similarity"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
        )

    assert not outputs_root.exists()
    assert not (cache_root / MERGE_STATE_DIRECTORY).exists()


def test_keep_existing_rejects_destination_similarity_with_wrong_identity(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "repo")
    model_identity = {
        "backend": "sentence-transformers",
        "embedding_representation": "normalized-float32",
        "local_files_only": True,
        "model": "fixture/minilm",
        "model_revision": "revision",
        "score_kind": "cosine-similarity",
    }
    identity = build_similarity_cache_identity(
        model_identity=model_identity,
        texts=("first", "second"),
    )
    SimilarityCache(config.parent / "cache").store(identity, ((1.0, 0.25), (0.25, 1.0)))
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "destination-cache").resolve()
    wrong_identity = build_similarity_cache_identity(
        model_identity=model_identity,
        texts=("second", "first"),
    )
    wrong = SimilarityCache(tmp_path / "wrong-cache").store(
        wrong_identity, ((1.0, 0.5), (0.5, 1.0))
    )
    destination = SimilarityCache(cache_root).entry_path(identity)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(wrong.read_bytes())
    outputs_root = (tmp_path / "destination-outputs").resolve()

    with pytest.raises(CacheBundleIntegrityError, match="similarity"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=outputs_root,
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
        )

    assert not outputs_root.exists()
    assert not (cache_root / MERGE_STATE_DIRECTORY).exists()


def test_keep_existing_never_overrides_immutable_conflicts(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path / "repo")
    bundle = (tmp_path / "bundle").resolve()
    pack_bundle(config, "rag2026-0", bundle)
    cache_root = (tmp_path / "merged-cache").resolve()
    source_planning = _planning_entry(config.parent / "cache")
    conflicting = cache_root / source_planning.relative_to(config.parent / "cache")
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"immutable destination")

    with pytest.raises(CacheBundleConflictError, match="immutable"):
        merge_bundles(
            cache_root=cache_root,
            outputs_root=(tmp_path / "merged-outputs").resolve(),
            bundle_dirs=(bundle,),
            score_conflicts="keep-existing",
        )
