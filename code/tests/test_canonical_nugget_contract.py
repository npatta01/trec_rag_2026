from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from trec_rag.canonical_nuggets import (
    CanonicalArtifacts,
    OpenRouterCanonicalNuggetBackend,
    build_canonical_nugget_request,
    canonicalize_subnarrative,
    run_canonical_stage,
)
from trec_rag.nuggetizer_adapter import NuggetizerCanonicalNuggetBackend
from trec_rag.evidence_store import _selection_json
from trec_rag.facet_evidence import (
    BudgetSnapshot,
    EvidenceMember,
    SelectionPolicy,
    SemanticCluster,
    SubnarrativeContext,
    SubnarrativeSelection,
)
from trec_rag.facet_extraction import (
    BackendReply,
    FacetResponse,
    OPENROUTER_DEEPSEEK_MODEL,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _selection(index: int = 1) -> SubnarrativeSelection:
    evidence = (
        EvidenceMember(
            candidate_nugget_id=f"n{index}-a",
            candidate_kind="exact_sentence",
            text=f"Exact evidence {index} opened the center in 2020.",
            docid=f"doc-{index}-a",
            document_sha256="a" * 64,
            raw_logit=9.0,
        ),
        EvidenceMember(
            candidate_nugget_id=f"n{index}-b",
            candidate_kind="exact_sentence",
            text=f"Exact evidence {index} recorded ten shelters.",
            docid=f"doc-{index}-b",
            document_sha256="b" * 64,
            raw_logit=8.0,
        ),
    )
    cluster = SemanticCluster(
        cluster_id=f"cluster-{index}",
        representative_candidate_nugget_id=evidence[0].candidate_nugget_id,
        representative_text=evidence[0].text,
        representative_raw_logit=evidence[0].raw_logit,
        members=evidence,
        supports=evidence,
        support_document_count=2,
    )
    return SubnarrativeSelection(
        schema_version="subnarrative_selection_v1",
        context=SubnarrativeContext(
            topic_id="224",
            official_narrative="Explain documented coastal safety measures.",
            subnarrative_id=f"subnarrative-{index}",
            subnarrative_text=f"Coastal safety aspect {index}",
        ),
        policy=SelectionPolicy(budgets=(1,), precluster_limit=10),
        similarity_identity=(("model", "literal-test-matrix"),),
        candidate_count=2,
        exact_group_count=2,
        precluster_count=2,
        semantic_cluster_count=1,
        clusters=(cluster,),
        snapshots=(BudgetSnapshot(1, (cluster.cluster_id,), False),),
    )


def _write_inputs(tmp_path: Path, *, count: int = 1) -> tuple[Path, Path]:
    selections_path = tmp_path / "subnarrative-selections.jsonl"
    rows = [_selection_json(_selection(index)) for index in range(1, count + 1)]
    selections_bytes = b"".join(_canonical(row) + b"\n" for row in rows)
    selections_path.write_bytes(selections_bytes)
    manifest_path = tmp_path / "selection-manifest.json"
    manifest_path.write_bytes(
        _canonical(
            {
                "schema_version": "subnarrative_selection_manifest_v1",
                "selection_schema_version": "subnarrative_selection_v1",
                "context_schema_version": "subnarrative_selection_context_v1",
                "candidate_schema_version": "extractive_candidate_nugget_v1",
                "candidate_manifest_schema_version": "extractive_candidate_manifest_v1",
                "candidates_file": "candidates.jsonl",
                "candidate_manifest_file": "candidate-manifest.json",
                "contexts_file": "selection-contexts.jsonl",
                "selection_file": selections_path.name,
                "candidates_sha256": "a" * 64,
                "candidate_manifest_sha256": "b" * 64,
                "contexts_sha256": "c" * 64,
                "selections_sha256": sha256(selections_bytes).hexdigest(),
                "output_sha256": sha256(selections_bytes).hexdigest(),
                "candidate_rows_scanned": count * 2,
                "candidate_projection_count": count * 2,
                "loaded_candidate_projection_count": count * 2,
                "context_count": count,
                "selection_count": count,
                "exact_group_count": count * 2,
                "semantic_cluster_count": count,
                "selected_cluster_count": count,
                "policy": {
                    "budgets": [1],
                    "precluster_limit": 10,
                    "semantic_threshold": 0.92,
                    "mmr_lambda": 0.7,
                },
                "embedding_identity": {"model": "literal-test-matrix"},
                "candidate_scorer_identity": {
                    "model": "fake",
                    "model_revision": "pin",
                    "backend_version": "test",
                    "score_representation": "raw_logits",
                    "inference_dtype": "float32",
                    "score_kind": "extractive_sentence_v1",
                    "sentence_max_length": 512,
                },
                "retrieval_network_calls": 0,
                "hosted_llm_calls": 0,
            }
        )
        + b"\n"
    )
    return selections_path, manifest_path


def _stage(
    tmp_path: Path,
    selections_path: Path,
    selection_manifest_path: Path,
    **kwargs: object,
) -> CanonicalArtifacts:
    return run_canonical_stage(
        selections_path=selections_path,
        selection_manifest_path=selection_manifest_path,
        selected_budget=1,
        output_path=tmp_path / "canonical-nuggets.jsonl",
        manifest_path=tmp_path / "canonical-nugget-manifest.json",
        cache_dir=tmp_path / "response-cache",
        cache_ignore_checker=lambda _path: True,
        **kwargs,
    )


def _metadata() -> dict[str, object]:
    return {
        "requested_model": OPENROUTER_DEEPSEEK_MODEL,
        "response_model": OPENROUTER_DEEPSEEK_MODEL,
        "provider": "test-provider",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class _Backend:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.calls: list[object] = []

    def complete(self, request: object) -> BackendReply:
        self.calls.append(request)
        return BackendReply(
            content=self.content,
            response_body=b'{"safe":"raw response"}',
            status=200,
            metadata=_metadata(),
        )


def test_stage_keeps_claims_evidence_bound_and_binds_configured_limits(
    tmp_path: Path,
) -> None:
    """Catches accepting model evidence text/IDs or ignoring configured caps."""
    selections, selection_manifest = _write_inputs(tmp_path)
    backend = _Backend(
        b'{"claims":[{"claim":"The center opened in 2020.",'
        b'"evidence_aliases":["e001"]}]}'
    )

    artifacts = _stage(
        tmp_path,
        selections,
        selection_manifest,
        max_canonical_claims=1,
        max_supporting_documents_per_claim=1,
        backend_factory=lambda: backend,
    )

    assert artifacts == CanonicalArtifacts(
        nuggets_path=tmp_path / "canonical-nuggets.jsonl",
        manifest_path=tmp_path / "canonical-nugget-manifest.json",
    )
    assert len(backend.calls) == 1
    request = backend.calls[0]
    assert request.max_canonical_claims == 1
    assert request.max_supporting_documents_per_claim == 1
    payload = json.loads(request.request_body)
    prompt_limits = json.loads(payload["messages"][-1]["content"])["output"]
    assert prompt_limits["max_claims"] == 1
    assert prompt_limits["max_supporting_documents_per_claim"] == 1
    result = json.loads(artifacts.nuggets_path.read_bytes())
    assert result["nuggets"][0]["claim_text"] == "The center opened in 2020."
    assert result["nuggets"][0]["evidence"] == [
        {
            "candidate_nugget_id": "n1-a",
            "candidate_kind": "exact_sentence",
            "text": "Exact evidence 1 opened the center in 2020.",
            "text_sha256": sha256(
                b"Exact evidence 1 opened the center in 2020."
            ).hexdigest(),
            "docid": "doc-1-a",
            "document_sha256": "a" * 64,
            "cluster_id": "cluster-1",
        }
    ]
    manifest = json.loads(artifacts.manifest_path.read_bytes())
    assert manifest["schema_version"] == "canonical_nugget_manifest_v2"
    assert manifest["max_canonical_claims"] == 1
    assert manifest["max_supporting_documents_per_claim"] == 1
    assert manifest["request_sha256s"] == [request.request_sha256]
    assert manifest["hosted_llm_calls"] == 1
    raw_cache = (
        tmp_path / "response-cache" / "raw" / f"{request.request_sha256}.json"
    )
    validated_cache = (
        tmp_path
        / "response-cache"
        / "validated"
        / f"{request.request_sha256}.json"
    )
    assert json.loads(raw_cache.read_bytes())["schema_version"] == (
        "canonical_nugget_raw_cache_v1"
    )
    assert json.loads(validated_cache.read_bytes())["schema_version"] == (
        "canonical_nugget_validated_cache_v2"
    )


def test_empty_stage_never_constructs_a_backend(tmp_path: Path) -> None:
    """Catches paying for hosted work when upstream selected no evidence."""
    selections, selection_manifest = _write_inputs(tmp_path, count=0)
    constructed: list[bool] = []

    artifacts = _stage(
        tmp_path,
        selections,
        selection_manifest,
        backend_factory=lambda: constructed.append(True),
    )

    assert constructed == []
    assert artifacts.nuggets_path.read_bytes() == b""
    manifest = json.loads(artifacts.manifest_path.read_bytes())
    assert manifest["result_count"] == 0
    assert manifest["hosted_llm_calls"] == 0


def test_empty_stage_rejects_boolean_budget_before_persisting(tmp_path: Path) -> None:
    """Catches a Boolean budget bypass when there are no requests to validate it."""
    selections, selection_manifest = _write_inputs(tmp_path, count=0)

    with pytest.raises(TypeError, match="selected_budget"):
        run_canonical_stage(
            selections_path=selections,
            selection_manifest_path=selection_manifest,
            selected_budget=True,
            output_path=tmp_path / "canonical-nuggets.jsonl",
            manifest_path=tmp_path / "canonical-nugget-manifest.json",
            cache_dir=tmp_path / "response-cache",
            cache_ignore_checker=lambda _path: True,
        )

    assert not (tmp_path / "canonical-nuggets.jsonl").exists()


@pytest.mark.parametrize(
    "content",
    (
        b'{"claims":',
        b'{"claims":[{"claim":"Unsupported.","evidence_aliases":["e999"]}]}',
        b'{"claims":[{"claim":"Two documents.","evidence_aliases":["e001","e002"]}]}',
    ),
    ids=("malformed", "unknown-evidence", "support-cap"),
)
def test_one_shot_failure_uses_exact_extractive_fallback(
    tmp_path: Path,
    content: bytes,
) -> None:
    """Catches retries, partial model acceptance, and invented fallback text."""
    selections, selection_manifest = _write_inputs(tmp_path)
    backend = _Backend(content)

    artifacts = _stage(
        tmp_path,
        selections,
        selection_manifest,
        max_supporting_documents_per_claim=1,
        backend_factory=lambda: backend,
    )

    assert len(backend.calls) == 1
    row = json.loads(artifacts.nuggets_path.read_bytes())
    assert row["state"] == "fallback_extractive"
    assert row["nuggets"][0]["claim_text"] == (
        "Exact evidence 1 opened the center in 2020."
    )
    assert row["nuggets"][0]["nugget_kind"] == "extractive_fallback"
    assert row["nuggets"][0]["evidence"][0]["text"] == row["nuggets"][0]["claim_text"]


def test_validated_cache_is_revalidated_against_the_current_request(
    tmp_path: Path,
) -> None:
    """Catches treating a hash-consistent but semantically invalid cache as trusted."""
    selections, selection_manifest = _write_inputs(tmp_path)
    safe = _Backend(
        b'{"claims":[{"claim":"Safe cached claim.","evidence_aliases":["e001"]}]}'
    )
    artifacts = _stage(
        tmp_path,
        selections,
        selection_manifest,
        backend_factory=lambda: safe,
    )
    first_bytes = artifacts.nuggets_path.read_bytes()
    validated = next((tmp_path / "response-cache" / "validated").glob("*.json"))
    cached = json.loads(validated.read_bytes())
    cached["content"] = (
        '{"claims":[{"claim":"Stale.","evidence_aliases":["e999"]}]}'
    )
    cached["content_sha256"] = sha256(cached["content"].encode()).hexdigest()
    validated.write_bytes(_canonical(cached) + b"\n")
    artifacts.nuggets_path.unlink()
    artifacts.manifest_path.unlink()

    recovered = _stage(
        tmp_path,
        selections,
        selection_manifest,
        backend_factory=lambda: (_ for _ in ()).throw(
            AssertionError("a valid raw reply must prevent another hosted call")
        ),
    )

    assert recovered.nuggets_path.read_bytes() == first_bytes
    rewritten = json.loads(validated.read_bytes())
    assert "e999" not in rewritten["content"]
    assert json.loads(recovered.manifest_path.read_bytes())["hosted_llm_calls"] == 0


def test_selection_manifest_tamper_is_rejected_before_backend_construction(
    tmp_path: Path,
) -> None:
    """Catches hosted calls made from a re-signed but false upstream count."""
    selections, selection_manifest = _write_inputs(tmp_path)
    manifest = json.loads(selection_manifest.read_bytes())
    manifest["exact_group_count"] = 999
    selection_manifest.write_bytes(_canonical(manifest) + b"\n")
    constructed: list[bool] = []

    with pytest.raises(ValueError, match="exact_group_count"):
        _stage(
            tmp_path,
            selections,
            selection_manifest,
            backend_factory=lambda: constructed.append(True),
        )

    assert constructed == []


@pytest.mark.parametrize("mutation", ("id", "kind", "claim"))
def test_warm_resume_rejects_canonical_result_semantic_tamper_before_backend(
    tmp_path: Path,
    mutation: str,
) -> None:
    selections, selection_manifest = _write_inputs(tmp_path)
    artifacts = _stage(
        tmp_path,
        selections,
        selection_manifest,
        backend_factory=lambda: _Backend(
            b'{"claims":[{"claim":"The center opened in 2020.",'
            b'"evidence_aliases":["e001"]}]}'
        ),
    )
    row = json.loads(artifacts.nuggets_path.read_bytes())
    nugget = row["nuggets"][0]
    if mutation == "id":
        nugget["canonical_nugget_id"] = "canonical-forged"
    elif mutation == "kind":
        nugget["nugget_kind"] = "extractive_fallback"
    else:
        nugget["claim_text"] = "Altered but re-signed claim."
    output_bytes = _canonical(row) + b"\n"
    artifacts.nuggets_path.write_bytes(output_bytes)
    manifest = json.loads(artifacts.manifest_path.read_bytes())
    digest = sha256(output_bytes).hexdigest()
    manifest["canonical_nuggets_sha256"] = digest
    manifest["output_sha256"] = digest
    artifacts.manifest_path.write_bytes(_canonical(manifest) + b"\n")
    constructed: list[bool] = []

    with pytest.raises(ValueError, match="existing manifest"):
        _stage(
            tmp_path,
            selections,
            selection_manifest,
            backend_factory=lambda: constructed.append(True),
        )

    assert constructed == []


def test_credential_reflection_is_absent_from_all_persisted_artifacts(
    tmp_path: Path,
) -> None:
    """Catches decoded credentials reaching caches, results, or manifests."""
    selections, selection_manifest = _write_inputs(tmp_path)
    content = (
        r'{"claims":[{"claim":"\u0073ecret-test-key","evidence_aliases":["e001"]}]}'
    )
    envelope = _canonical(
        {
            "id": "generation-1",
            "object": "chat.completion",
            "created": 1,
            "model": OPENROUTER_DEEPSEEK_MODEL,
            "provider": "test-provider",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )

    class _Transport:
        def send(self, _request: object) -> FacetResponse:
            return FacetResponse(200, envelope)

    artifacts = _stage(
        tmp_path,
        selections,
        selection_manifest,
        backend_factory=lambda: OpenRouterCanonicalNuggetBackend(
            environ={"OPENROUTER_API_KEY": "secret-test-key"},
            transport=_Transport(),
        ),
    )

    persisted = artifacts.nuggets_path.read_bytes() + artifacts.manifest_path.read_bytes()
    persisted += b"".join(
        path.read_bytes()
        for path in (tmp_path / "response-cache").rglob("*")
        if path.is_file()
    )
    assert b"secret-test-key" not in persisted
    assert json.loads(artifacts.nuggets_path.read_bytes())["state"] == "fallback_extractive"


def test_empty_request_never_calls_backend() -> None:
    """Catches a lower-level hosted call for an empty validated snapshot."""
    selection = _selection()
    empty = replace(
        selection,
        candidate_count=0,
        exact_group_count=0,
        precluster_count=0,
        semantic_cluster_count=0,
        clusters=(),
        snapshots=(BudgetSnapshot(1, (), True),),
    )
    request = build_canonical_nugget_request(empty, 1)

    result = canonicalize_subnarrative(request, object())

    assert result.state == "empty"
    assert result.backend_attempts == 0


def _openrouter_envelope(content: str) -> bytes:
    return _canonical(
        {
            "id": "generation-1",
            "object": "chat.completion",
            "created": 1,
            "model": OPENROUTER_DEEPSEEK_MODEL,
            "provider": "test-provider",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def test_nuggetizer_adapter_sends_one_grounded_creator_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches loss or reordering of sealed evidence at the package boundary."""
    from nuggetizer.models.nuggetizer import Nuggetizer

    request = build_canonical_nugget_request(_selection(), 1)
    captured_requests: list[object] = []
    original_create = Nuggetizer.create

    def capture_create(self: object, package_request: object) -> object:
        captured_requests.append(package_request)
        return original_create(self, package_request)

    monkeypatch.setattr(Nuggetizer, "create", capture_create)

    class _Transport:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def send(self, sent: object) -> FacetResponse:
            self.requests.append(sent)
            return FacetResponse(
                200,
                _openrouter_envelope(
                    '{"claims":[{"claim":"The center opened in 2020.",'
                    '"evidence_aliases":["e001"]}]}'
                ),
            )

    transport = _Transport()
    result = canonicalize_subnarrative(
        request,
        NuggetizerCanonicalNuggetBackend(
            environ={"OPENROUTER_API_KEY": "test-key"}, transport=transport
        ),
    )

    assert len(captured_requests) == 1
    package_request = captured_requests[0]
    assert package_request.query.text == "Coastal safety aspect 1"
    assert [document.docid for document in package_request.documents] == ["n1-a", "n1-b"]
    assert [document.title for document in package_request.documents] == [None, None]
    assert [document.segment for document in package_request.documents] == [
        "e001: Exact evidence 1 opened the center in 2020.",
        "e002: Exact evidence 1 recorded ten shelters.",
    ]
    assert len(transport.requests) == 1
    assert transport.requests[0].body == request.request_body
    assert result.state == "complete"
    assert result.nuggets[0].claim_text == "The center opened in 2020."
    assert result.nuggets[0].evidence[0].candidate_nugget_id == "n1-a"


def test_nuggetizer_adapter_translates_the_package_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches bypassing the package result after the creator response is valid."""
    from nuggetizer.models.nuggetizer import Nuggetizer

    request = build_canonical_nugget_request(_selection(), 1)
    original_create = Nuggetizer.create

    def discard_created_nuggets(self: object, package_request: object) -> object:
        original_create(self, package_request)
        return []

    monkeypatch.setattr(Nuggetizer, "create", discard_created_nuggets)

    class _Transport:
        def send(self, _sent: object) -> FacetResponse:
            return FacetResponse(
                200,
                _openrouter_envelope(
                    '{"claims":[{"claim":"The center opened in 2020.",'
                    '"evidence_aliases":["e001"]}]}'
                ),
            )

    result = canonicalize_subnarrative(
        request,
        NuggetizerCanonicalNuggetBackend(
            environ={"OPENROUTER_API_KEY": "test-key"}, transport=_Transport()
        ),
    )

    assert result.state == "fallback_extractive"
    assert tuple(nugget.claim_text for nugget in result.nuggets) == tuple(
        row.text for row in request.fallback_evidence
    )


@pytest.mark.parametrize(
    ("failure", "response", "wire_calls"),
    (
        ("package", None, 0),
        ("unknown_alias", '{"claims":[{"claim":"Unsupported.","evidence_aliases":["e999"]}]}', 1),
        ("malformed", '{"claims":', 1),
    ),
)
def test_nuggetizer_adapter_fails_closed_without_a_package_retry(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    response: str | None,
    wire_calls: int,
) -> None:
    """Catches hidden package failures being accepted as an empty model result."""
    from nuggetizer.models.nuggetizer import Nuggetizer

    request = build_canonical_nugget_request(_selection(), 1)
    package_calls: list[object] = []
    original_create = Nuggetizer.create

    def capture_create(self: object, package_request: object) -> object:
        package_calls.append(package_request)
        if failure == "package":
            raise RuntimeError("package create failed")
        return original_create(self, package_request)

    monkeypatch.setattr(Nuggetizer, "create", capture_create)

    class _Transport:
        def __init__(self) -> None:
            self.calls = 0

        def send(self, _sent: object) -> FacetResponse:
            self.calls += 1
            assert response is not None
            return FacetResponse(200, _openrouter_envelope(response))

    transport = _Transport()
    result = canonicalize_subnarrative(
        request,
        NuggetizerCanonicalNuggetBackend(
            environ={"OPENROUTER_API_KEY": "test-key"}, transport=transport
        ),
    )

    assert len(package_calls) == 1
    assert transport.calls == wire_calls
    assert result.state == "fallback_extractive"
    assert tuple(nugget.claim_text for nugget in result.nuggets) == tuple(
        row.text for row in request.fallback_evidence
    )


@pytest.mark.parametrize(
    ("failure", "response", "expected_hosted_calls"),
    (
        ("package", None, 0),
        ("malformed", '{"claims":', 1),
        (
            "success",
            '{"claims":[{"claim":"The center opened in 2020.",'
            '"evidence_aliases":["e001"]}]}',
            1,
        ),
    ),
)
def test_stage_manifest_counts_actual_nuggetizer_transport_invocations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    response: str | None,
    expected_hosted_calls: int,
) -> None:
    """Catches backend attempts being reported as hosted transport calls."""
    from nuggetizer.models.nuggetizer import Nuggetizer

    selections, selection_manifest = _write_inputs(tmp_path)
    original_create = Nuggetizer.create

    def fail_before_transport(self: object, package_request: object) -> object:
        if failure == "package":
            raise RuntimeError("package create failed")
        return original_create(self, package_request)

    monkeypatch.setattr(Nuggetizer, "create", fail_before_transport)

    class _Transport:
        def __init__(self) -> None:
            self.calls = 0

        def send(self, _sent: object) -> FacetResponse:
            self.calls += 1
            assert response is not None
            return FacetResponse(200, _openrouter_envelope(response))

    transport = _Transport()
    artifacts = _stage(
        tmp_path,
        selections,
        selection_manifest,
        backend_factory=lambda: NuggetizerCanonicalNuggetBackend(
            environ={"OPENROUTER_API_KEY": "test-key"}, transport=transport
        ),
    )

    manifest = json.loads(artifacts.manifest_path.read_bytes())
    assert transport.calls == expected_hosted_calls
    assert manifest["hosted_llm_calls"] == expected_hosted_calls
