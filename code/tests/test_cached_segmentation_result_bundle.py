from __future__ import annotations

import copy
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import pytest
import zstandard

from trec_rag.facet_extraction import BackendReply
from trec_rag.generation_handoff import (
    ClaimHint,
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
from trec_rag.retrieval_nugget_coverage import (
    CoverageModelRequest,
    CoverageRunConfig,
    run_coverage_evaluation,
)


TOPICS = ("14", "31")
NARRATIVE = "Explain topic evidence."
PLAN = {
    "schema_version": "retrieval_nugget_plan_v1",
    "facets": [
        {
            "title": "Evidence",
            "obligations": [
                {
                    "requirement": "Explain topic evidence.",
                    "support_test": "Evidence is present.",
                    "kind": "required_explicit",
                    "narrative_spans": ["topic evidence"],
                }
            ],
        }
    ],
    "unmapped_narrative_spans": [],
}
JUDGMENT = {
    "schema_version": "retrieval_nugget_judgment_v1",
    "judgments": [
        {
            "obligation_id": "f001-o001",
            "label": "full",
            "supporting_nugget_aliases": ["n001"],
            "missing_elements": "",
        }
    ],
}


class _Backend:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def complete(self, request: CoverageModelRequest) -> BackendReply:
        return BackendReply(
            content=json.dumps(self.payload, separators=(",", ":")).encode(),
            response_body=b'{"provider":"fixture"}',
            status=200,
            metadata={"requested_model": request.model},
        )


def _digest(value: str | bytes) -> str:
    body = value.encode() if isinstance(value, str) else value
    return sha256(body).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _write_json(path: Path, value: object) -> bytes:
    body = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def _with_content_digest(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["receipt_content_sha256"] = _digest(_canonical(result))
    return result


def _handoff(path: Path, *, topics: tuple[str, ...] = TOPICS) -> Path:
    rows: list[GenerationTopic] = []
    for topic_id in topics:
        text = f"Evidence for topic {topic_id}."
        evidence_id = f"evidence-{topic_id}"
        evidence = EvidencePassage(
            evidence_id=evidence_id,
            group_id="evidence",
            cluster_id="cluster-1",
            cluster_ordinal=1,
            support_ordinal=1,
            candidate_kind="exact_sentence",
            docid=f"doc-{topic_id}",
            document_rank=1,
            text=text,
            document_sha256=_digest(text),
            source_span=EvidenceSourceSpan(0, len(text), 0, len(text.encode())),
        )
        rows.append(
            GenerationTopic(
                topic_id,
                NARRATIVE,
                (
                    EvidenceGroup(
                        "evidence",
                        "generated_subnarrative",
                        "Topic evidence",
                        (
                            SelectedCluster(
                                "cluster-1", 1, evidence_id, (evidence_id,)
                            ),
                        ),
                    ),
                ),
                (evidence,),
                (
                    ClaimHint(
                        f"claim-{topic_id}",
                        "evidence",
                        "canonical",
                        text,
                        (evidence_id,),
                    ),
                ),
                TopicSourceReceipts("a" * 64, "b" * 64),
            )
        )
    write_generation_handoff(
        path,
        GenerationHandoff(
            HandoffProducer("topic_records_v4", "fixture-run", "c" * 40),
            tuple(rows),
        ),
    )
    return path


def _baseline_fixture(tmp_path: Path) -> tuple[Path, Path]:
    handoff = _handoff(tmp_path / "generation_handoff_manifest.json")
    coverage_root = tmp_path / "retrieval_nugget_coverage_v2"
    for topic_id in TOPICS:
        run_coverage_evaluation(
            CoverageRunConfig(
                handoff_manifest_path=handoff,
                topic_id=topic_id,
                work_dir=coverage_root / topic_id,
                allow_hosted_calls=True,
            ),
            planner=_Backend(PLAN),
            judge=_Backend(JUDGMENT),
        )
    return handoff, coverage_root


def _write_phase(topic_root: Path, topic_id: str, phase: str) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    receipts = []
    for relative in sorted(bundle_module.PHASE_ARTIFACTS[phase]):
        body = (
            b"sqlite-fixture"
            if relative == "records.sqlite3"
            else _canonical({"artifact": relative, "topic_id": topic_id})
        )
        path = topic_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        receipts.append(
            {"relative_path": relative, "bytes": len(body), "sha256": _digest(body)}
        )
    _write_json(
        topic_root / phase / "complete.json",
        {"topic_id": topic_id, "artifacts": receipts},
    )


def _result_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    validation_root = tmp_path / "validation"
    handoff = _handoff(run_root / "generation_handoff_manifest.json")
    config_sha = "d" * 64
    projection_digests: list[str] = []
    receipt_digests: list[str] = []
    topic_receipts: list[dict[str, str]] = []
    empty_stages = {
        stage: {
            "cache_hits": 1,
            "cache_misses": 0,
            "network_calls": 0,
            "provider_calls": 0,
            "model_batches": 0,
        }
        for stage in (
            "planning",
            "retrieval",
            "passage_scores",
            "sentence_scores",
            "similarity",
            "canonicalization",
        )
    }
    for topic_id in TOPICS:
        topic_root = run_root / topic_id
        result = _write_json(
            topic_root / "decomposition" / "result.json",
            {"topic_id": topic_id, "narrative": NARRATIVE},
        )
        _write_json(
            topic_root / "decomposition" / "manifest.json",
            {
                "result_file": "result.json",
                "result_bytes": len(result),
                "result_sha256": _digest(result),
            },
        )
        for phase in ("retrieval", "scoring", "canonical"):
            _write_phase(topic_root, topic_id, phase)
        projection_body = (
            topic_root / "canonical" / "retrieval-projection-manifest.json"
        ).read_bytes()
        projection_sha = _digest(projection_body)
        projection_digests.append(projection_sha)
        topic_receipts.append(
            {"topic_id": topic_id, "projection_manifest_sha256": projection_sha}
        )
        operation = _with_content_digest(
            {
                "schema_version": "cache-operation-receipt-v1",
                "mode": "cached-upstream-rescore",
                "run_id": "fixture-run",
                "topic_id": topic_id,
                "config_sha256": config_sha,
                "projection_manifest_sha256": projection_sha,
                "phases": {
                    name: {"resumed": False}
                    for name in ("planning", "retrieval", "scoring", "canonical")
                },
                "stages": empty_stages,
            }
        )
        operation_body = _write_json(
            topic_root / "cache-operation-receipt.cached-upstream-rescore.json",
            operation,
        )
        receipt_digests.append(_digest(operation_body))
        _write_json(
            topic_root / "topic-job-receipt.cached-upstream-rescore.json",
            {
                "schema_version": "topic-job-receipt-v4",
                "run_id": "fixture-run",
                "topic_id": topic_id,
                "config_sha256": config_sha,
                "mode": "cached-upstream-rescore",
                "projection_manifest_sha256": projection_sha,
                "status": "complete",
                "stopping_reason": "coverage_sufficient",
            },
        )

    official = b"14 Q0 doc-1 1 1 fixture\n31 Q0 doc-2 1 1 fixture\n"
    (run_root / "r_output_trec_rag_2026.tsv").write_bytes(official)
    archive = b"PK\x03\x04fixture"
    (run_root / "retrieval_with_text.jsonl.zip").write_bytes(archive)
    handoff_body = handoff.read_bytes()
    export_body = _write_json(
        run_root / "retrieval_export_manifest.json",
        {
            "schema_version": "retrieval_export_manifest_v6",
            "run_id": "fixture-run",
            "export_code_commit": "c" * 40,
            "selected_topic_ids": list(TOPICS),
            "execution": {"topic_workers": 4},
            "topic_receipts": topic_receipts,
            "artifacts": {
                "generation_handoff_manifest.json": {
                    "bytes": len(handoff_body),
                    "sha256": _digest(handoff_body),
                },
                "r_output_trec_rag_2026.tsv": {
                    "bytes": len(official),
                    "sha256": _digest(official),
                },
                "retrieval_with_text.jsonl.zip": {
                    "bytes": len(archive),
                    "sha256": _digest(archive),
                },
            },
        },
    )
    totals = {
        stage: {
            counter: value * len(TOPICS)
            for counter, value in counters.items()
        }
        for stage, counters in empty_stages.items()
    }
    _write_json(
        run_root / "cache-operation-manifest.json",
        _with_content_digest(
            {
                "schema_version": "cache-operation-manifest-v1",
                "mode": "cached-upstream-rescore",
                "run_id": "fixture-run",
                "config_sha256": config_sha,
                "topic_ids": list(TOPICS),
                "topic_receipt_sha256s": receipt_digests,
                "projection_manifest_sha256s": projection_digests,
                "retrieval_export_manifest": "retrieval_export_manifest.json",
                "retrieval_export_manifest_sha256": _digest(export_body),
                "totals": totals,
            }
        ),
    )

    coverage_root = validation_root / "retrieval_nugget_coverage_v2"
    for topic_id in TOPICS:
        run_coverage_evaluation(
            CoverageRunConfig(
                handoff_manifest_path=handoff,
                topic_id=topic_id,
                work_dir=coverage_root / topic_id,
                allow_hosted_calls=True,
            ),
            planner=_Backend(PLAN),
            judge=_Backend(JUDGMENT),
        )
    handoff_sha = _digest(handoff_body)
    for kind in ("structural", "semantic"):
        comparison = {
            "schema_version": f"cached-segmentation-{kind}-comparison-v1",
            "baseline_handoff_sha256": "e" * 64,
            "candidate_handoff_sha256": handoff_sha,
            "topic_ids": list(TOPICS),
            "gates_passed": True,
        }
        if kind == "semantic":
            comparison.update(
                {"planner_calls": 0, "candidate_judge_calls": len(TOPICS)}
            )
        else:
            comparison["improvement_observed"] = True
        comparison_body = _write_json(
            validation_root / f"{kind}-comparison.json", comparison
        )
        _write_json(
            validation_root / f"{kind}-comparison-manifest.json",
            {
                "schema_version": f"cached-segmentation-{kind}-manifest-v1",
                "comparison_file": f"{kind}-comparison.json",
                "comparison_bytes": len(comparison_body),
                "comparison_sha256": _digest(comparison_body),
                "topic_ids": list(TOPICS),
                "gates_passed": True,
            },
        )
    _write_json(
        validation_root / "concurrency-decision.json",
        {
            "schema_version": "cached-segmentation-concurrency-decision-v1",
            "run_id": "fixture-run",
            "config_sha256": config_sha,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
            "driver_version": "575.57.08",
            "probe_topics": ["14", "31"],
            "probe_workers": 2,
            "selected_workers": 4,
            "elapsed_seconds": 120,
            "idle_memory_mib": 1000,
            "peak_memory_mib": 20000,
            "total_memory_mib": 81559,
            "projected_four_worker_memory_mib": 39000,
            "safe_limit_memory_mib": 73403,
        },
    )
    (run_root / ".retrieval-export.lock").write_text("excluded")
    return run_root, validation_root


def _rewrite_archive(bundle: Path, mutation: str) -> None:
    archive_path = bundle / "bundle.tar.zst"
    with zstandard.ZstdDecompressor().stream_reader(
        io.BytesIO(archive_path.read_bytes())
    ) as reader:
        raw = reader.read()
    if mutation == "trailing":
        rewritten = raw + b"undeclared trailing bytes"
    else:
        members: list[tuple[tarfile.TarInfo, bytes | None]] = []
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            for info in archive:
                body = archive.extractfile(info)
                members.append((copy.copy(info), None if body is None else body.read()))
        target = members[1][0]
        if mutation == "traversal":
            target.name = "../escape"
        elif mutation == "absolute":
            target.name = "/absolute"
        elif mutation == "symlink":
            target.type = tarfile.SYMTYPE
            target.linkname = "target"
            target.size = 0
            members[1] = (target, None)
        elif mutation == "device":
            target.type = tarfile.CHRTYPE
            target.size = 0
            members[1] = (target, None)
        elif mutation in {"duplicate", "collision"}:
            extra_info = copy.copy(members[-1][0])
            if mutation == "collision":
                extra_info.name = extra_info.name.swapcase()
            members.append((extra_info, members[-1][1]))
        else:  # pragma: no cover - test helper contract
            raise AssertionError(mutation)
        sink = io.BytesIO()
        with tarfile.open(fileobj=sink, mode="w:", format=tarfile.GNU_FORMAT) as archive:
            for info, body in members:
                archive.addfile(info, None if body is None else io.BytesIO(body))
        rewritten = sink.getvalue()
    compressed = zstandard.ZstdCompressor(
        level=10, write_checksum=True, write_content_size=True
    ).compress(rewritten)
    archive_path.write_bytes(compressed)
    marker_path = bundle / "bundle-complete.json"
    marker = json.loads(marker_path.read_text())
    marker["archive_size"] = len(compressed)
    marker["archive_sha256"] = _digest(compressed)
    marker_path.write_bytes(_canonical(marker))


def test_baseline_bundle_roundtrip_is_deterministic_and_revalidates_each_topic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    handoff, coverage_root = _baseline_fixture(tmp_path / "source")
    first = tmp_path / "first"
    second = tmp_path / "second"
    bundle_module.pack_baseline_bundle(handoff, coverage_root, first)
    bundle_module.pack_baseline_bundle(handoff, coverage_root, second)
    assert (first / "bundle.tar.zst").read_bytes() == (
        second / "bundle.tar.zst"
    ).read_bytes()

    calls: list[str] = []
    original = bundle_module.load_completed_coverage_evaluation

    def recording_loader(**kwargs):
        calls.append(kwargs["topic_id"])
        return original(**kwargs)

    monkeypatch.setattr(
        bundle_module, "load_completed_coverage_evaluation", recording_loader
    )
    verified = bundle_module.verify_baseline_bundle(first)

    assert verified.bundle_kind == "baseline"
    assert verified.topic_ids == TOPICS
    assert calls == list(TOPICS)
    assert {path.name for path in first.iterdir()} == {
        "bundle.tar.zst",
        "bundle-complete.json",
    }
    restored = tmp_path / "restored"
    bundle_module.restore_baseline_bundle(first, restored)
    assert (restored / "generation_handoff_manifest.json").is_file()
    assert sorted((restored / "retrieval_nugget_coverage_v2").iterdir()) == [
        restored / "retrieval_nugget_coverage_v2" / topic_id
        for topic_id in TOPICS
    ]


def test_baseline_pack_rejects_missing_completed_topic(tmp_path: Path) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    handoff, coverage_root = _baseline_fixture(tmp_path / "source")
    (coverage_root / TOPICS[-1] / "manifest.json").unlink()

    with pytest.raises(bundle_module.ResultBundleIntegrityError, match="coverage"):
        bundle_module.pack_baseline_bundle(
            handoff,
            coverage_root,
            tmp_path / "bundle",
        )


def test_result_bundle_roundtrip_binds_run_revision_topics_and_excludes_locks(
    tmp_path: Path,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root, validation_root = _result_fixture(tmp_path / "source")
    bundle = tmp_path / "bundle"
    packed = bundle_module.pack_result_bundle(run_root, validation_root, bundle)
    verified = bundle_module.verify_result_bundle(bundle)

    assert packed == verified
    assert verified.bundle_kind == "result"
    assert verified.run_id == "fixture-run"
    assert verified.git_revision == "c" * 40
    assert verified.topic_ids == TOPICS
    assert all(".lock" not in member.path for member in verified.members)
    assert any(member.path.endswith("records.sqlite3") for member in verified.members)
    assert any("semantic-comparison.json" in member.path for member in verified.members)
    assert any("concurrency-decision.json" in member.path for member in verified.members)


def test_diagnostic_bundle_preserves_failed_semantic_evidence_and_is_nonpromotable(
    tmp_path: Path,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root, validation_root = _result_fixture(tmp_path / "source")
    comparison_path = validation_root / "semantic-comparison.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["gates_passed"] = False
    comparison_body = _write_json(comparison_path, comparison)
    manifest_path = validation_root / "semantic-comparison-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["comparison_bytes"] = len(comparison_body)
    manifest["comparison_sha256"] = _digest(comparison_body)
    manifest["gates_passed"] = False
    _write_json(manifest_path, manifest)

    bundle = tmp_path / "diagnostic"
    packed = bundle_module.pack_diagnostic_bundle(
        run_root,
        validation_root,
        bundle,
        run_id="fixture-run",
        revision="c" * 40,
        topic_ids=TOPICS,
        stage="semantic-validation",
        exit_status=2,
    )
    verified = bundle_module.verify_diagnostic_bundle(bundle)

    assert packed == verified
    assert verified.bundle_kind == "diagnostic"
    assert any(
        member.path == "validation/semantic-comparison.json"
        for member in verified.members
    )
    assert any(
        member.path.endswith("retrieval_nugget_coverage_v2/14/judgments.json")
        for member in verified.members
    )
    with pytest.raises(bundle_module.ResultBundleIntegrityError, match="kind"):
        bundle_module.verify_result_bundle(bundle)


def test_diagnostic_bundle_tolerates_partial_interruption_and_keeps_completed_phase(
    tmp_path: Path,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root = tmp_path / "source" / "run"
    validation_root = tmp_path / "source" / "validation"
    _handoff(run_root / "generation_handoff_manifest.json", topics=("14",))
    _write_phase(run_root / "14", "14", "retrieval")
    validation_root.mkdir(parents=True)

    bundle = tmp_path / "diagnostic"
    bundle_module.pack_diagnostic_bundle(
        run_root,
        validation_root,
        bundle,
        run_id="fixture-run",
        revision="c" * 40,
        topic_ids=TOPICS,
        stage="concurrency-probe",
        exit_status=137,
    )
    verified = bundle_module.verify_diagnostic_bundle(bundle)

    paths = {member.path for member in verified.members}
    assert "diagnostic-status.json" in paths
    assert "run/generation_handoff_manifest.json" in paths
    assert "run/14/retrieval/complete.json" in paths
    assert "run/14/retrieval/evidence-bundle.json" in paths
    assert all(not path.startswith("run/31/") for path in paths)


def test_diagnostic_bundle_preserves_canary_comparison_after_handoff_expansion(
    tmp_path: Path,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root, validation_root = _result_fixture(tmp_path / "source")
    canary = validation_root / "canary"
    comparison = json.loads(
        (validation_root / "structural-comparison.json").read_text()
    )
    comparison["topic_ids"] = ["14"]
    comparison["candidate_handoff_sha256"] = "f" * 64
    comparison_body = _write_json(canary / "structural-comparison.json", comparison)
    manifest = json.loads(
        (validation_root / "structural-comparison-manifest.json").read_text()
    )
    manifest["topic_ids"] = ["14"]
    manifest["comparison_bytes"] = len(comparison_body)
    manifest["comparison_sha256"] = _digest(comparison_body)
    _write_json(canary / "structural-comparison-manifest.json", manifest)

    bundle = tmp_path / "diagnostic"
    bundle_module.pack_diagnostic_bundle(
        run_root,
        validation_root,
        bundle,
        run_id="fixture-run",
        revision="c" * 40,
        topic_ids=TOPICS,
        stage="semantic-validation",
        exit_status=2,
    )
    verified = bundle_module.verify_diagnostic_bundle(bundle)

    assert any(
        member.path == "validation/canary/structural-comparison.json"
        for member in verified.members
    )


def test_diagnostic_bundle_requires_a_failure_exit_status(tmp_path: Path) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root, validation_root = _result_fixture(tmp_path / "source")
    with pytest.raises(ValueError, match="failure"):
        bundle_module.pack_diagnostic_bundle(
            run_root,
            validation_root,
            tmp_path / "diagnostic",
            run_id="fixture-run",
            revision="c" * 40,
            topic_ids=TOPICS,
            stage="semantic-validation",
            exit_status=0,
        )


def test_diagnostic_bundle_skips_an_interrupted_unsealed_comparison(
    tmp_path: Path,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root = tmp_path / "source" / "run"
    validation_root = tmp_path / "source" / "validation"
    run_root.mkdir(parents=True)
    _write_json(
        validation_root / "semantic-comparison.json",
        {
            "schema_version": "cached-segmentation-semantic-comparison-v1",
            "candidate_handoff_sha256": "f" * 64,
            "topic_ids": list(TOPICS),
            "gates_passed": False,
        },
    )

    bundle = tmp_path / "diagnostic"
    bundle_module.pack_diagnostic_bundle(
        run_root,
        validation_root,
        bundle,
        run_id="fixture-run",
        revision="c" * 40,
        topic_ids=TOPICS,
        stage="semantic-validation",
        exit_status=143,
    )
    verified = bundle_module.verify_diagnostic_bundle(bundle)

    assert {member.path for member in verified.members} == {
        "diagnostic-status.json"
    }


@pytest.mark.parametrize("field", ["run_id", "export_code_commit"])
def test_result_pack_rejects_changed_run_or_git_identity(
    tmp_path: Path,
    field: str,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root, validation_root = _result_fixture(tmp_path / "source")
    path = run_root / "retrieval_export_manifest.json"
    value = json.loads(path.read_text())
    value[field] = "changed"
    path.write_bytes(_canonical(value))

    with pytest.raises(bundle_module.ResultBundleIntegrityError, match="export identity"):
        bundle_module.pack_result_bundle(
            run_root, validation_root, tmp_path / "bundle"
        )


def test_verify_rejects_tampered_archive_and_bundle_directory_extras(
    tmp_path: Path,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    handoff, coverage_root = _baseline_fixture(tmp_path / "source")
    bundle = tmp_path / "bundle"
    bundle_module.pack_baseline_bundle(handoff, coverage_root, bundle)
    archive = bundle / "bundle.tar.zst"
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(bundle_module.ResultBundleIntegrityError, match="archive"):
        bundle_module.verify_baseline_bundle(bundle)

    clean = tmp_path / "clean"
    bundle_module.pack_baseline_bundle(handoff, coverage_root, clean)
    (clean / "extra").write_text("undeclared")
    with pytest.raises(bundle_module.ResultBundleIntegrityError, match="undeclared"):
        bundle_module.verify_baseline_bundle(clean)


@pytest.mark.parametrize(
    "mutation",
    ["traversal", "absolute", "symlink", "device", "duplicate", "collision"],
)
def test_verify_rejects_hostile_archive_members(
    tmp_path: Path,
    mutation: str,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    handoff, coverage_root = _baseline_fixture(tmp_path / "source")
    bundle = tmp_path / "bundle"
    bundle_module.pack_baseline_bundle(handoff, coverage_root, bundle)
    _rewrite_archive(bundle, mutation)

    with pytest.raises(bundle_module.ResultBundleIntegrityError):
        bundle_module.verify_baseline_bundle(bundle)


def test_verify_rejects_trailing_decompressed_payload(tmp_path: Path) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    handoff, coverage_root = _baseline_fixture(tmp_path / "source")
    bundle = tmp_path / "bundle"
    bundle_module.pack_baseline_bundle(handoff, coverage_root, bundle)
    _rewrite_archive(bundle, "trailing")

    with pytest.raises(bundle_module.ResultBundleIntegrityError, match="trailing"):
        bundle_module.verify_baseline_bundle(bundle)


def test_result_pack_rejects_wrong_topic_order_and_missing_comparison_manifest(
    tmp_path: Path,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root, validation_root = _result_fixture(tmp_path / "wrong-order")
    operation_path = run_root / "cache-operation-manifest.json"
    operation = json.loads(operation_path.read_text())
    operation["topic_ids"] = list(reversed(TOPICS))
    operation.pop("receipt_content_sha256")
    operation_path.write_bytes(_canonical(_with_content_digest(operation)))
    with pytest.raises(bundle_module.ResultBundleIntegrityError, match="operation manifest"):
        bundle_module.pack_result_bundle(
            run_root, validation_root, tmp_path / "wrong-order-bundle"
        )

    run_root, validation_root = _result_fixture(tmp_path / "missing-manifest")
    (validation_root / "semantic-comparison-manifest.json").unlink()
    with pytest.raises(bundle_module.ResultBundleIntegrityError, match="manifest"):
        bundle_module.pack_result_bundle(
            run_root, validation_root, tmp_path / "missing-manifest-bundle"
        )


def test_result_pack_rejects_structural_pass_without_observed_improvement(
    tmp_path: Path,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root, validation_root = _result_fixture(tmp_path / "source")
    comparison_path = validation_root / "structural-comparison.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["improvement_observed"] = False
    comparison_body = _write_json(comparison_path, comparison)
    manifest_path = validation_root / "structural-comparison-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["comparison_bytes"] = len(comparison_body)
    manifest["comparison_sha256"] = _digest(comparison_body)
    _write_json(manifest_path, manifest)

    with pytest.raises(
        bundle_module.ResultBundleIntegrityError,
        match="structural comparison identity",
    ):
        bundle_module.pack_result_bundle(
            run_root, validation_root, tmp_path / "bundle"
        )


def test_result_pack_rejects_unsafe_concurrency_decision(tmp_path: Path) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root, validation_root = _result_fixture(tmp_path / "source")
    path = validation_root / "concurrency-decision.json"
    value = json.loads(path.read_text())
    value["projected_four_worker_memory_mib"] = 80000
    _write_json(path, value)

    with pytest.raises(
        bundle_module.ResultBundleIntegrityError,
        match="concurrency decision",
    ):
        bundle_module.pack_result_bundle(
            run_root, validation_root, tmp_path / "bundle"
        )


def test_result_pack_rejects_internally_inconsistent_concurrency_decision(
    tmp_path: Path,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root, validation_root = _result_fixture(tmp_path / "source")
    path = validation_root / "concurrency-decision.json"
    value = json.loads(path.read_text())
    value["projected_four_worker_memory_mib"] = 0
    value["safe_limit_memory_mib"] = 0
    _write_json(path, value)

    with pytest.raises(
        bundle_module.ResultBundleIntegrityError,
        match="concurrency decision",
    ):
        bundle_module.pack_result_bundle(
            run_root, validation_root, tmp_path / "bundle"
        )


def test_result_pack_rejects_concurrency_decision_from_another_run(
    tmp_path: Path,
) -> None:
    import trec_rag.cached_segmentation_result_bundle as bundle_module

    run_root, validation_root = _result_fixture(tmp_path / "source")
    path = validation_root / "concurrency-decision.json"
    value = json.loads(path.read_text())
    value["run_id"] = "different-run"
    _write_json(path, value)

    with pytest.raises(
        bundle_module.ResultBundleIntegrityError,
        match="concurrency decision",
    ):
        bundle_module.pack_result_bundle(
            run_root, validation_root, tmp_path / "bundle"
        )
