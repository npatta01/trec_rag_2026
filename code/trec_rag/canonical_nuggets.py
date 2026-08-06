"""One-call, provenance-linked canonical nuggetization for one subnarrative."""

from __future__ import annotations

import base64
import binascii
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Protocol

from trec_rag.evidence_store import decode_subnarrative_selection
from trec_rag.facet_evidence import (
    EvidenceMember,
    SCORING_NORMALIZATION_VERSION,
    SELECTION_SCHEMA_VERSION,
    SelectionPolicy,
    SubnarrativeSelection,
)
from trec_rag.facet_extraction import (
    BackendReply,
    FacetRequest,
    FacetResponse,
    MAX_RESPONSE_BYTES,
    OPENROUTER_BASE_URL,
    OPENROUTER_DEEPSEEK_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    _UrllibFacetTransport,
    _decode_json,
    _openrouter_completion,
)
from trec_rag.topic_records import CANDIDATE_STAGE, TOPIC_RECORDS_SCHEMA_VERSION


CANONICAL_NUGGET_SCHEMA_VERSION = "canonical_nuggets_v2"
PROMPT_VERSION = "canonical_nuggetizer_v5"
MAX_CANONICAL_NUGGETS = 20
MAX_EVIDENCE_ALIASES_PER_CLAIM = 3
MAX_SUPPORTING_DOCUMENTS_PER_CLAIM = 3
MAX_CANONICAL_REQUEST_BYTES = 1_000_000
MAX_CLAIM_CHARACTERS = 1_000
SELECTION_MANIFEST_SCHEMA_VERSION = "subnarrative_selection_manifest_v1"
RESULT_SCHEMA_VERSION = "canonical_nugget_result_v2"
MANIFEST_SCHEMA_VERSION = "canonical_nugget_manifest_v2"
RAW_CACHE_SCHEMA_VERSION = "canonical_nugget_raw_cache_v2"
VALIDATED_CACHE_SCHEMA_VERSION = "canonical_nugget_validated_cache_v3"
NUGGET_IMPORTANCE_VALUES = frozenset({"vital", "okay"})
SCORER_MODE_VALUES = frozenset({"hosted", "local_all_okay"})

_SAFE_METADATA_FIELDS = frozenset({
    "requested_model", "response_model", "provider", "finish_reason", "usage",
})
_SAFE_USAGE_FIELDS = frozenset({
    "prompt_tokens", "completion_tokens", "total_tokens", "cost", "is_byok",
    "completion_tokens_details", "cost_details", "prompt_tokens_details",
    "server_tool_use_details",
})
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_MANIFEST_FIELDS = frozenset({
    "schema_version", "selection_schema_version", "context_schema_version",
    "candidate_schema_version", "records_schema_version", "records_stage",
    "records_file", "records_manifest_file", "contexts_file", "selection_file",
    "records_database_sha256", "candidate_semantic_sha256", "contexts_sha256",
    "selections_sha256", "output_sha256",
    "candidate_rows_scanned", "candidate_projection_count", "loaded_candidate_projection_count",
    "context_count", "selection_count", "exact_group_count", "semantic_cluster_count",
    "selected_cluster_count", "policy", "embedding_identity", "candidate_scorer_identity",
    "retrieval_network_calls", "hosted_llm_calls",
})
_MANIFEST_FIELDS = frozenset({
    "schema_version", "result_schema_version", "canonical_response_schema_version",
    "selection_schema_version", "selection_manifest_schema_version", "selection_file",
    "selection_manifest_file", "canonical_nugget_file", "selections_sha256",
    "selection_manifest_sha256", "canonical_nuggets_sha256", "output_sha256",
    "selected_budget", "selection_count", "result_count", "state_counts",
    "max_canonical_claims", "max_supporting_documents_per_claim",
    "request_sha256s", "model", "prompt_version", "hosted_llm_calls",
    "validated_cache_hits", "raw_cache_writes", "validated_cache_writes",
})
_RESULT_FIELDS = frozenset({
    "schema_version", "topic_id", "subnarrative_id", "selected_budget",
    "request_sha256", "state", "nuggets", "metadata", "error",
})
_RAW_CACHE_FIELDS = frozenset({
    "schema_version", "request_sha256", "status", "content_base64",
    "response_bodies_base64", "response_body_sha256s", "metadata",
    "metadata_entries",
})
_RUN_COUNTER_FIELDS = (
    "hosted_llm_calls", "validated_cache_hits", "raw_cache_writes",
    "validated_cache_writes",
)
_SCORER_IDENTITY_FIELDS = frozenset({
    "model", "model_revision", "backend_version", "score_representation",
    "inference_dtype", "score_kind", "sentence_max_length", "input_policy",
})
_RESULT_STATES = frozenset({"complete", "empty", "fallback_extractive"})


@dataclass(frozen=True)
class CanonicalEvidence:
    """Authoritative exact evidence resolved in Python from a compact alias."""

    alias: str
    cluster_alias: str
    cluster_id: str
    candidate_nugget_id: str
    candidate_kind: str
    text: str
    text_sha256: str
    docid: str
    document_sha256: str


@dataclass(frozen=True)
class CanonicalNuggetRequest:
    """One bounded request plus all exact local state needed to validate it."""

    topic_id: str
    subnarrative_id: str
    subnarrative_text: str
    selected_budget: int
    max_canonical_claims: int
    max_supporting_documents_per_claim: int
    evidence: tuple[CanonicalEvidence, ...]
    fallback_evidence: tuple[CanonicalEvidence, ...]
    request_body: bytes
    request_sha256: str
    scorer_mode: str = "hosted"


@dataclass(frozen=True)
class CanonicalNugget:
    """A model claim or extractive fallback with separately retained evidence."""

    canonical_nugget_id: str
    claim_text: str
    importance: str
    evidence: tuple[CanonicalEvidence, ...]


@dataclass(frozen=True)
class CanonicalNuggetResult:
    topic_id: str
    subnarrative_id: str
    selected_budget: int
    request_sha256: str
    state: str
    nuggets: tuple[CanonicalNugget, ...]
    metadata: Mapping[str, object]
    backend_attempts: int
    error: str | None = None


@dataclass(frozen=True)
class CanonicalArtifacts:
    """Sealed canonical result and manifest paths returned by the typed stage."""

    nuggets_path: Path
    manifest_path: Path


class CanonicalNuggetBackend(Protocol):
    def complete(self, request: CanonicalNuggetRequest) -> BackendReply: ...


class CanonicalValidatedCacheError(RuntimeError):
    """Base class for fail-closed validated canonical cache reads."""


class CanonicalValidatedCacheMiss(CanonicalValidatedCacheError):
    """The exact validated canonical response is absent."""


class CanonicalValidatedCacheIntegrityError(CanonicalValidatedCacheError):
    """A validated canonical response is malformed or semantically invalid."""


class OpenRouterCanonicalNuggetBackend:
    """One-shot bounded OpenRouter adapter for the fixed canonical request."""

    one_shot_no_retry = True
    redirects_allowed = False

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        transport: object | None = None,
    ) -> None:
        import os

        key = (os.environ if environ is None else environ).get("OPENROUTER_API_KEY")
        if not isinstance(key, str) or not key:
            raise ValueError("OPENROUTER_API_KEY must be set to non-empty text")
        self._key = key
        self._transport = _UrllibFacetTransport() if transport is None else transport
        self._transport_invocation_count = 0

    @property
    def transport_invocation_count(self) -> int:
        """Return exact calls made through the hosted transport boundary."""
        return self._transport_invocation_count

    def complete(self, request: CanonicalNuggetRequest) -> BackendReply:
        if not isinstance(request, CanonicalNuggetRequest):
            raise TypeError("request must be CanonicalNuggetRequest")
        sent = FacetRequest(
            url=OPENROUTER_BASE_URL + "/chat/completions",
            body=request.request_body,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Metadata": "enabled",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        try:
            send = getattr(self._transport, "send", None)
            if not callable(send):
                raise TypeError("transport must provide send(request)")
            self._transport_invocation_count += 1
            response = send(sent)
            if not isinstance(response, FacetResponse):
                raise TypeError("transport must return FacetResponse")
            if not 200 <= response.status < 300:
                raise ValueError(f"OpenRouter returned HTTP {response.status}")
            if len(response.body) > MAX_RESPONSE_BYTES:
                raise ValueError("OpenRouter response exceeded byte cap")
            envelope = _decode_json(
                response.body, "OpenRouter chat-completions envelope"
            )
            if _contains_credential(envelope, self._key):
                raise ValueError("OpenRouter response contained the API credential")
            completion = _openrouter_completion(envelope)
            content = _decode_json(
                completion.content.encode("utf-8"), "canonical nugget response"
            )
            if _contains_credential(content, self._key):
                raise ValueError("OpenRouter response contained the API credential")
            metadata = {
                "requested_model": OPENROUTER_DEEPSEEK_MODEL,
                "response_model": completion.response_model,
                "provider": completion.provider,
                "finish_reason": completion.finish_reason,
                "usage": dict(completion.usage),
            }
            return BackendReply(
                content=completion.content.encode("utf-8"),
                response_body=response.body,
                status=response.status,
                metadata=metadata,
                response_bodies=(response.body,),
                metadata_entries=(metadata,),
            )
        except Exception as exc:
            message = str(exc).replace(self._key, "[REDACTED]")
            if message == str(exc):
                raise
            raise RuntimeError(message) from None


def _contains_credential(value: object, credential: str) -> bool:
    if isinstance(value, str):
        return credential in value
    if isinstance(value, Mapping):
        return any(
            _contains_credential(key, credential)
            or _contains_credential(item, credential)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_credential(item, credential) for item in value)
    return False


def run_canonical_stage(
    *,
    selections_path: Path,
    selection_manifest_path: Path,
    selected_budget: int,
    output_path: Path,
    manifest_path: Path,
    cache_dir: Path,
    max_canonical_claims: int = MAX_CANONICAL_NUGGETS,
    max_supporting_documents_per_claim: int = MAX_SUPPORTING_DOCUMENTS_PER_CLAIM,
    scorer_mode: str = "hosted",
    backend_factory: Callable[[], object] | None = None,
    cache_ignore_checker: Callable[[Path], bool] | None = None,
    cache_only: bool = False,
    cache_stats: dict[str, int] | None = None,
) -> CanonicalArtifacts:
    """Canonicalize sealed selections through typed paths and settings."""
    if isinstance(selected_budget, bool) or not isinstance(selected_budget, int):
        raise TypeError("selected_budget must be a non-Boolean integer")
    if scorer_mode not in SCORER_MODE_VALUES:
        raise ValueError("scorer_mode must be 'hosted' or 'local_all_okay'")
    if not isinstance(cache_only, bool):
        raise TypeError("cache_only must be Boolean")
    if cache_stats is not None and not isinstance(cache_stats, dict):
        raise TypeError("cache_stats must be a dict")
    if cache_stats is not None:
        cache_stats.update(cache_hits=0, cache_misses=0, provider_calls=0)
    paths = tuple(
        _typed_path(value, label)
        for value, label in (
            (selections_path, "selections_path"),
            (selection_manifest_path, "selection_manifest_path"),
            (output_path, "output_path"),
            (manifest_path, "manifest_path"),
        )
    )
    selections_path, selection_manifest_path, output_path, manifest_path = paths
    cache_dir = _typed_path(cache_dir, "cache_dir").resolve()
    resolved_paths = tuple(path.resolve() for path in paths)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("all input and output paths must name different files")
    if any(path == cache_dir or cache_dir in path.parents for path in resolved_paths):
        raise ValueError("input and output files must be outside the cache directory")
    if cache_dir.exists() and not cache_dir.is_dir():
        raise ValueError("cache path must be a directory")
    if backend_factory is not None and not callable(backend_factory):
        raise TypeError("backend_factory must be callable")
    if cache_ignore_checker is not None and not callable(cache_ignore_checker):
        raise TypeError("cache_ignore_checker must be callable")
    _validate_limit(
        max_canonical_claims,
        "max_canonical_claims",
        maximum=MAX_CANONICAL_NUGGETS,
    )
    _validate_limit(
        max_supporting_documents_per_claim,
        "max_supporting_documents_per_claim",
        maximum=MAX_SUPPORTING_DOCUMENTS_PER_CLAIM,
    )
    _require_safe_cache_root(cache_dir, cache_ignore_checker)

    (
        selections,
        selection_manifest,
        selection_policy,
    ) = load_validated_selection_artifacts(
        selections_path,
        selection_manifest_path,
    )
    selection_bytes = selections_path.read_bytes()
    selection_manifest_bytes = selection_manifest_path.read_bytes()
    if selected_budget not in selection_policy.budgets:
        raise ValueError("selected budget is absent from one or more selection snapshots")

    selections_sha256 = sha256(selection_bytes).hexdigest()
    selection_manifest_sha256 = sha256(selection_manifest_bytes).hexdigest()
    binding = {
        "selection_file": selections_path.name,
        "selection_manifest_file": selection_manifest_path.name,
        "canonical_nugget_file": output_path.name,
        "selections_sha256": selections_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "selected_budget": selected_budget,
        "max_canonical_claims": max_canonical_claims,
        "max_supporting_documents_per_claim": max_supporting_documents_per_claim,
    }
    requests = tuple(
        build_canonical_nugget_request(
            selection,
            selected_budget,
            max_canonical_claims=max_canonical_claims,
            max_supporting_documents_per_claim=max_supporting_documents_per_claim,
            scorer_mode=scorer_mode,
        )
        for selection in selections
    )
    existing_identity = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "canonical_response_schema_version": CANONICAL_NUGGET_SCHEMA_VERSION,
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "selection_manifest_schema_version": SELECTION_MANIFEST_SCHEMA_VERSION,
        **binding,
        "selection_count": len(selections),
        "result_count": len(requests),
        "request_sha256s": [request.request_sha256 for request in requests],
        "model": OPENROUTER_DEEPSEEK_MODEL,
        "prompt_version": PROMPT_VERSION,
    }
    artifacts = CanonicalArtifacts(output_path, manifest_path)
    existing_run = _validate_existing_run(
        output_path, manifest_path, existing_identity, requests
    )
    if existing_run and not cache_only:
        return artifacts

    live_backend: object | None = None
    results: list[CanonicalNuggetResult] = []
    hosted_calls = 0
    validated_hits = 0
    raw_writes = 0
    validated_writes = 0
    for request in requests:
        validated_path = cache_dir / "validated" / f"{request.request_sha256}.json"
        raw_path = cache_dir / "raw" / f"{request.request_sha256}.json"
        if cache_only:
            if request.evidence and cache_stats is not None:
                cache_stats["cache_misses"] += 1
            result = require_validated_canonical_result(cache_dir, request)
            validated_hits += int(bool(request.evidence))
            if request.evidence and cache_stats is not None:
                cache_stats["cache_misses"] -= 1
                cache_stats["cache_hits"] += 1
            validate_canonical_nugget_result(_result_json(result), request)
            results.append(result)
            continue
        cached = _load_validated(validated_path, request) if validated_path.is_file() else None
        result: CanonicalNuggetResult | None = None
        if cached is not None:
            cached_result = canonicalize_subnarrative(
                request,
                _FixedReplyBackend(cached),
            )
            if cached_result.state == "complete":
                result = replace(cached_result, backend_attempts=0)
                validated_hits += 1
        if result is None and not request.evidence:
            result = canonicalize_subnarrative(request, _NoBackend())
        elif result is None:
            raw_reply = _load_raw(raw_path, request) if raw_path.is_file() else None
            if raw_reply is not None:
                raw_result = canonicalize_subnarrative(
                    request,
                    _FixedReplyBackend(raw_reply),
                )
                result = replace(raw_result, backend_attempts=0)
                if result.state == "complete":
                    _write_validated(validated_path, request, raw_reply)
                    validated_writes += 1
            else:
                if live_backend is None:
                    if backend_factory is None:
                        from trec_rag.nuggetizer_adapter import (
                            NuggetizerCanonicalNuggetBackend,
                        )

                        if scorer_mode == "hosted":
                            factory = NuggetizerCanonicalNuggetBackend
                        else:
                            factory = lambda: NuggetizerCanonicalNuggetBackend(
                                scorer_mode=scorer_mode
                            )
                    else:
                        factory = backend_factory
                    live_backend = factory()
                caching_backend = _RawCachingBackend(
                    live_backend,
                    raw_path,
                    request.request_sha256,
                )
                transport_calls_before = _transport_invocation_count(live_backend)
                result = canonicalize_subnarrative(request, caching_backend)
                transport_calls_after = _transport_invocation_count(live_backend)
                if (
                    transport_calls_before is not None
                    and transport_calls_after is not None
                    and transport_calls_after >= transport_calls_before
                ):
                    hosted_calls += transport_calls_after - transport_calls_before
                else:
                    hosted_calls += result.backend_attempts
                raw_writes += int(caching_backend.wrote_raw)
                if result.state == "complete" and caching_backend.reply is not None:
                    _write_validated(validated_path, request, caching_backend.reply)
                    validated_writes += 1
        if result is None:  # pragma: no cover - defensive branch invariant
            raise RuntimeError("canonical nugget result was not resolved")
        validate_canonical_nugget_result(_result_json(result), request)
        results.append(result)

    if cache_stats is not None and not cache_only:
        required = sum(bool(request.evidence) for request in requests)
        cache_stats.update(
            cache_hits=validated_hits,
            cache_misses=required - validated_hits,
            provider_calls=hosted_calls,
        )

    if existing_run:
        return artifacts

    output_bytes = b"".join(
        _canonical_json(_result_json(result)) + b"\n" for result in results
    )
    output_sha256 = sha256(output_bytes).hexdigest()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "canonical_response_schema_version": CANONICAL_NUGGET_SCHEMA_VERSION,
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "selection_manifest_schema_version": SELECTION_MANIFEST_SCHEMA_VERSION,
        **binding,
        "canonical_nuggets_sha256": output_sha256,
        "output_sha256": output_sha256,
        "selection_count": len(selections),
        "result_count": len(results),
        "state_counts": dict(
            sorted(Counter(result.state for result in results).items())
        ),
        "request_sha256s": [request.request_sha256 for request in requests],
        "model": OPENROUTER_DEEPSEEK_MODEL,
        "prompt_version": PROMPT_VERSION,
        "hosted_llm_calls": hosted_calls,
        "validated_cache_hits": validated_hits,
        "raw_cache_writes": raw_writes,
        "validated_cache_writes": validated_writes,
    }
    _atomic_write(output_path, output_bytes)
    _atomic_write(manifest_path, _canonical_json(manifest) + b"\n")
    return artifacts


def _typed_path(value: Path, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be a Path")
    return value


def build_canonical_nugget_request(
    selection: SubnarrativeSelection,
    budget: int,
    *,
    max_canonical_claims: int = MAX_CANONICAL_NUGGETS,
    max_supporting_documents_per_claim: int = MAX_SUPPORTING_DOCUMENTS_PER_CLAIM,
    scorer_mode: str = "hosted",
    max_request_bytes: int = MAX_CANONICAL_REQUEST_BYTES,
) -> CanonicalNuggetRequest:
    """Build one fixed, bounded request from one validated selection snapshot."""
    if not isinstance(selection, SubnarrativeSelection):
        raise TypeError("selection must be a validated SubnarrativeSelection")
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget must be a non-Boolean integer")
    if isinstance(max_request_bytes, bool) or not isinstance(max_request_bytes, int) or max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive")
    _validate_limit(
        max_canonical_claims,
        "max_canonical_claims",
        maximum=MAX_CANONICAL_NUGGETS,
    )
    _validate_limit(
        max_supporting_documents_per_claim,
        "max_supporting_documents_per_claim",
        maximum=MAX_SUPPORTING_DOCUMENTS_PER_CLAIM,
    )
    if scorer_mode not in SCORER_MODE_VALUES:
        raise ValueError("scorer_mode must be 'hosted' or 'local_all_okay'")
    snapshots = {snapshot.budget: snapshot for snapshot in selection.snapshots}
    snapshot = snapshots.get(budget)
    if snapshot is None:
        raise ValueError("budget must name one validated selection snapshot")
    cluster_by_id = {cluster.cluster_id: cluster for cluster in selection.clusters}
    evidence: list[CanonicalEvidence] = []
    fallbacks: list[CanonicalEvidence] = []
    seen_evidence: dict[str, CanonicalEvidence] = {}
    for cluster_index, cluster_id in enumerate(snapshot.cluster_ids, start=1):
        cluster = cluster_by_id[cluster_id]
        cluster_alias = f"c{cluster_index:03d}"
        selected_rows: list[CanonicalEvidence] = []
        for member in cluster.supports:
            row = _canonical_evidence(
                member,
                alias=f"e{len(evidence) + 1:03d}",
                cluster_alias=cluster_alias,
                cluster_id=cluster.cluster_id,
            )
            previous = seen_evidence.get(row.candidate_nugget_id)
            if previous is not None:
                if _evidence_identity(previous) != _evidence_identity(row):
                    raise ValueError("selection reuses an evidence ID with conflicting provenance")
                raise ValueError("selection reuses an evidence ID across selected clusters")
            seen_evidence[row.candidate_nugget_id] = row
            evidence.append(row)
            selected_rows.append(row)
        representative = next(
            row for row in selected_rows
            if row.candidate_nugget_id == cluster.representative_candidate_nugget_id
        )
        fallbacks.append(representative)

    from trec_rag.nuggetizer_adapter import render_nuggetizer_request_body

    request_body = render_nuggetizer_request_body(
        topic_id=selection.context.topic_id,
        subnarrative_id=selection.context.subnarrative_id,
        subnarrative_text=selection.context.subnarrative_text,
        evidence=tuple(evidence),
        max_canonical_claims=max_canonical_claims,
        max_supporting_documents_per_claim=max_supporting_documents_per_claim,
        scorer_mode=scorer_mode,
    )
    if len(request_body) > max_request_bytes:
        raise ValueError("canonical nugget request exceeds configured byte limit")
    return CanonicalNuggetRequest(
        topic_id=selection.context.topic_id,
        subnarrative_id=selection.context.subnarrative_id,
        subnarrative_text=selection.context.subnarrative_text,
        selected_budget=budget,
        max_canonical_claims=max_canonical_claims,
        max_supporting_documents_per_claim=max_supporting_documents_per_claim,
        evidence=tuple(evidence),
        fallback_evidence=tuple(fallbacks[:max_canonical_claims]),
        request_body=request_body,
        request_sha256=sha256(request_body).hexdigest(),
        scorer_mode=scorer_mode,
    )


def canonicalize_subnarrative(
    request: CanonicalNuggetRequest,
    backend: CanonicalNuggetBackend,
) -> CanonicalNuggetResult:
    """Complete at most once and fail closed to deterministic exact representatives."""
    if not isinstance(request, CanonicalNuggetRequest):
        raise TypeError("request must be CanonicalNuggetRequest")
    if not request.evidence:
        return _result(request, "empty", (), {}, 0, None)
    metadata: Mapping[str, object] = {}
    attempts = 0
    try:
        complete = getattr(backend, "complete", None)
        if not callable(complete):
            raise TypeError("backend must provide complete(request)")
        attempts = 1
        reply = complete(request)
        if not isinstance(reply, BackendReply):
            raise TypeError("backend must return BackendReply")
        if not 200 <= reply.status < 300:
            raise ValueError(f"canonical backend returned HTTP {reply.status}")
        if not isinstance(reply.content, bytes) or not isinstance(reply.response_body, bytes):
            raise TypeError("backend reply bodies must be bytes")
        if reply.response_bodies:
            response_bodies = _reply_response_bodies(reply)
            _reply_metadata_entries(reply, len(response_bodies))
            if any(len(body) > MAX_RESPONSE_BYTES for body in response_bodies):
                raise ValueError("canonical backend response exceeded byte cap")
        elif reply.metadata_entries and any(
            not isinstance(entry, Mapping) for entry in reply.metadata_entries
        ):
            raise TypeError("backend reply metadata provenance is invalid")
        elif len(reply.response_body) > MAX_RESPONSE_BYTES:
            raise ValueError("canonical backend response exceeded byte cap")
        metadata = _validated_metadata(reply.metadata)
        nuggets = _parse_claims(reply.content, request)
        return _result(request, "complete", nuggets, metadata, attempts, None)
    except Exception as exc:
        fallback = tuple(
            CanonicalNugget(
                canonical_nugget_id=_nugget_id(
                    request, row.text, (row.candidate_nugget_id,), "extractive"
                ),
                claim_text=row.text,
                importance="okay",
                evidence=(row,),
            )
            for row in request.fallback_evidence[:request.max_canonical_claims]
        )
        return _result(
            request,
            "fallback_extractive",
            fallback,
            metadata,
            attempts,
            _safe_error(exc),
        )


def _canonical_evidence(
    member: EvidenceMember,
    *,
    alias: str,
    cluster_alias: str,
    cluster_id: str,
) -> CanonicalEvidence:
    return CanonicalEvidence(
        alias=alias,
        cluster_alias=cluster_alias,
        cluster_id=cluster_id,
        candidate_nugget_id=member.candidate_nugget_id,
        candidate_kind=member.candidate_kind,
        text=member.text,
        text_sha256=sha256(member.text.encode("utf-8")).hexdigest(),
        docid=member.docid,
        document_sha256=member.document_sha256,
    )


def _evidence_identity(row: CanonicalEvidence) -> tuple[object, ...]:
    return (
        row.candidate_kind, row.text, row.text_sha256, row.docid, row.document_sha256,
    )


def _response_schema(
    aliases: tuple[str, ...],
    *,
    max_canonical_claims: int,
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "minItems": 0,
                "maxItems": max_canonical_claims,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["claim", "evidence_aliases"],
                    "properties": {
                        "claim": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_CLAIM_CHARACTERS,
                        },
                        "evidence_aliases": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_EVIDENCE_ALIASES_PER_CLAIM,
                            "items": {"type": "string", "enum": list(aliases)},
                        },
                    },
                },
            },
        },
    }


def _parse_claims(
    content: bytes,
    request: CanonicalNuggetRequest,
    *,
    allow_missing_importance: bool = False,
) -> tuple[CanonicalNugget, ...]:
    payload = _decode_json(content, "canonical nugget response")
    if set(payload) != {"claims"}:
        raise ValueError("canonical nugget response has unexpected or missing fields")
    claims = payload["claims"]
    if not isinstance(claims, list) or len(claims) > request.max_canonical_claims:
        raise ValueError("canonical nugget response exceeds configured claim cap")
    evidence_by_alias = {row.alias: row for row in request.evidence}
    evidence_order = {row.alias: index for index, row in enumerate(request.evidence)}
    seen_claims: set[str] = set()
    result: list[CanonicalNugget] = []
    for item in claims:
        if not isinstance(item, Mapping):
            raise ValueError("canonical claim has unexpected or missing fields")
        expected_fields = {"claim", "evidence_aliases"}
        if "importance" in item:
            expected_fields.add("importance")
        if set(item) != expected_fields or (
            not allow_missing_importance and "importance" not in item
        ):
            raise ValueError("canonical claim has unexpected or missing fields")
        claim = item["claim"]
        aliases = item["evidence_aliases"]
        importance = item.get("importance", "okay")
        if importance not in NUGGET_IMPORTANCE_VALUES:
            raise ValueError("canonical claim importance is invalid")
        if (
            not isinstance(claim, str)
            or not claim
            or claim != claim.strip()
            or len(claim) > MAX_CLAIM_CHARACTERS
            or "\n" in claim
            or "\r" in claim
        ):
            raise ValueError("canonical claim text is invalid")
        normalized = " ".join(claim.split()).casefold()
        if normalized in seen_claims:
            raise ValueError("canonical response contains duplicate claims")
        seen_claims.add(normalized)
        if (
            not isinstance(aliases, list)
            or not 1 <= len(aliases) <= MAX_EVIDENCE_ALIASES_PER_CLAIM
            or any(not isinstance(alias, str) or alias not in evidence_by_alias for alias in aliases)
            or len(set(aliases)) != len(aliases)
        ):
            raise ValueError("canonical claim evidence aliases are invalid")
        evidence = tuple(
            evidence_by_alias[alias]
            for alias in sorted(aliases, key=evidence_order.__getitem__)
        )
        if (
            len({row.docid for row in evidence})
            > request.max_supporting_documents_per_claim
        ):
            raise ValueError("canonical claim exceeds configured supporting document cap")
        evidence_ids = tuple(row.candidate_nugget_id for row in evidence)
        result.append(CanonicalNugget(
            canonical_nugget_id=_nugget_id(request, claim, evidence_ids, "model_claim"),
            claim_text=claim,
            importance=importance,
            evidence=evidence,
        ))
    return tuple(result)


def _validate_limit(value: int, label: str, *, maximum: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{label} must be within the supported range 1..{maximum}")


def _validated_metadata(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _SAFE_METADATA_FIELDS:
        raise ValueError("canonical backend completion metadata is invalid")
    requested = value["requested_model"]
    response_model = value["response_model"]
    provider = value["provider"]
    finish_reason = value["finish_reason"]
    usage = value["usage"]
    if requested != OPENROUTER_DEEPSEEK_MODEL:
        raise ValueError("canonical backend requested model is invalid")
    if not isinstance(response_model, str) or not response_model:
        raise ValueError("canonical backend response model is invalid")
    if provider is not None and (not isinstance(provider, str) or not provider):
        raise ValueError("canonical backend provider is invalid")
    if finish_reason != "stop":
        raise ValueError("canonical backend did not finish with stop")
    if not isinstance(usage, Mapping) or not {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(usage) or not set(usage) <= _SAFE_USAGE_FIELDS:
        raise ValueError("canonical backend usage is invalid")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage[key]) is not int or usage[key] < 0:
            raise ValueError("canonical backend usage token counts are invalid")
    for key in (
        "completion_tokens_details", "cost_details", "prompt_tokens_details",
        "server_tool_use_details",
    ):
        if key in usage and usage[key] is not None and not isinstance(usage[key], Mapping):
            raise ValueError("canonical backend usage details are invalid")
    safe_usage: dict[str, object] = {
        key: _safe_json_metadata(usage[key]) for key in sorted(usage)
    }
    if "cost" in safe_usage and safe_usage["cost"] is not None and (
        isinstance(safe_usage["cost"], bool) or not isinstance(safe_usage["cost"], (int, float))
    ):
        raise ValueError("canonical backend usage cost is invalid")
    if "is_byok" in safe_usage and safe_usage["is_byok"] is not None and type(safe_usage["is_byok"]) is not bool:
        raise ValueError("canonical backend usage byok flag is invalid")
    return {
        "requested_model": requested,
        "response_model": response_model,
        "provider": provider,
        "finish_reason": finish_reason,
        "usage": safe_usage,
    }


def _safe_json_metadata(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical backend metadata contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError("canonical backend metadata keys are invalid")
        return {key: _safe_json_metadata(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_safe_json_metadata(item) for item in value]
    raise ValueError("canonical backend metadata contains an invalid value")


def _nugget_id(
    request: CanonicalNuggetRequest,
    claim: str,
    evidence_ids: tuple[str, ...],
    kind: str,
) -> str:
    identity = _canonical_json({
        "topic_id": request.topic_id,
        "subnarrative_id": request.subnarrative_id,
        "kind": kind,
        "claim": claim,
        "evidence_ids": list(evidence_ids),
    })
    return "canonical-" + sha256(identity).hexdigest()[:24]


def validate_canonical_nugget_result(
    value: Mapping[str, object],
    request: CanonicalNuggetRequest,
) -> str:
    """Validate one persisted result against its exact producer request."""
    if not isinstance(request, CanonicalNuggetRequest):
        raise TypeError("request must be CanonicalNuggetRequest")
    if not isinstance(value, Mapping):
        raise ValueError("canonical nugget result must be an object")
    _require_fields(value, _RESULT_FIELDS, "canonical nugget result")
    state = value["state"]
    nuggets = value["nuggets"]
    metadata = value["metadata"]
    error = value["error"]
    if (
        value["schema_version"] != RESULT_SCHEMA_VERSION
        or value["topic_id"] != request.topic_id
        or value["subnarrative_id"] != request.subnarrative_id
        or type(value["selected_budget"]) is not int
        or value["selected_budget"] != request.selected_budget
        or value["request_sha256"] != request.request_sha256
        or type(state) is not str
        or state not in _RESULT_STATES
        or not isinstance(nuggets, list)
        or len(nuggets) > request.max_canonical_claims
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("canonical nugget result identity is invalid")
    if state == "empty":
        if request.evidence or nuggets or dict(metadata) or error is not None:
            raise ValueError("empty canonical nugget result semantics are invalid")
        return state
    if not request.evidence:
        raise ValueError("non-empty canonical result lacks request evidence")
    if state == "complete":
        if error is not None or dict(metadata) != _validated_metadata(metadata):
            raise ValueError("complete canonical nugget result state is invalid")
    else:
        if (
            type(error) is not str
            or not error
            or error != error.strip()
            or len(error) > 500
            or _CONTROL.search(error)
            or dict(metadata)
            and dict(metadata) != _validated_metadata(metadata)
        ):
            raise ValueError("fallback canonical nugget result state is invalid")

    evidence_by_id = {
        row.candidate_nugget_id: row for row in request.evidence
    }
    evidence_order = {
        row.candidate_nugget_id: index for index, row in enumerate(request.evidence)
    }
    fallback = request.fallback_evidence[: request.max_canonical_claims]
    if state == "fallback_extractive" and len(nuggets) != len(fallback):
        raise ValueError("fallback canonical nugget set is incomplete")
    seen_claims: set[str] = set()
    seen_ids: set[str] = set()
    for index, nugget in enumerate(nuggets):
        if not isinstance(nugget, Mapping):
            raise ValueError("canonical nugget must be an object")
        _require_fields(
            nugget,
            frozenset(
                {
                    "canonical_nugget_id",
                    "nugget_kind",
                    "claim_text",
                    "importance",
                    "evidence",
                }
            ),
            "canonical nugget",
        )
        kind = nugget["nugget_kind"]
        claim = nugget["claim_text"]
        importance = nugget["importance"]
        evidence = nugget["evidence"]
        expected_kind = (
            "model_claim" if state == "complete" else "extractive_fallback"
        )
        if (
            kind != expected_kind
            or type(claim) is not str
            or not claim
            or claim != claim.strip()
            or len(claim) > MAX_CLAIM_CHARACTERS
            or "\n" in claim
            or "\r" in claim
            or importance not in NUGGET_IMPORTANCE_VALUES
            or not isinstance(evidence, list)
            or not 1 <= len(evidence) <= MAX_EVIDENCE_ALIASES_PER_CLAIM
        ):
            raise ValueError("canonical nugget claim or kind is invalid")
        normalized = " ".join(claim.split()).casefold()
        if normalized in seen_claims:
            raise ValueError("canonical nugget claims are duplicated")
        seen_claims.add(normalized)

        evidence_rows: list[CanonicalEvidence] = []
        for item in evidence:
            if not isinstance(item, Mapping):
                raise ValueError("canonical nugget evidence must be an object")
            _require_fields(
                item,
                frozenset(
                    {
                        "candidate_nugget_id",
                        "candidate_kind",
                        "text",
                        "text_sha256",
                        "docid",
                        "document_sha256",
                        "cluster_id",
                    }
                ),
                "canonical nugget evidence",
            )
            candidate_id = item["candidate_nugget_id"]
            row = evidence_by_id.get(candidate_id) if isinstance(candidate_id, str) else None
            if row is None or dict(item) != _result_evidence_json(row):
                raise ValueError("canonical nugget evidence differs from its request")
            evidence_rows.append(row)
        evidence_ids = tuple(row.candidate_nugget_id for row in evidence_rows)
        if (
            len(set(evidence_ids)) != len(evidence_ids)
            or list(evidence_ids)
            != sorted(evidence_ids, key=evidence_order.__getitem__)
            or len({row.docid for row in evidence_rows})
            > request.max_supporting_documents_per_claim
        ):
            raise ValueError("canonical nugget evidence constraints are invalid")
        id_kind = "model_claim"
        if state == "fallback_extractive":
            expected = fallback[index]
            if evidence_rows != [expected] or claim != expected.text:
                raise ValueError("extractive fallback differs from its request")
            id_kind = "extractive"
        expected_id = _nugget_id(request, claim, evidence_ids, id_kind)
        nugget_id = nugget["canonical_nugget_id"]
        if nugget_id != expected_id or nugget_id in seen_ids:
            raise ValueError("canonical nugget ID is invalid or duplicated")
        seen_ids.add(nugget_id)
    return state


def _result(
    request: CanonicalNuggetRequest,
    state: str,
    nuggets: tuple[CanonicalNugget, ...],
    metadata: Mapping[str, object],
    attempts: int,
    error: str | None,
) -> CanonicalNuggetResult:
    return CanonicalNuggetResult(
        topic_id=request.topic_id,
        subnarrative_id=request.subnarrative_id,
        selected_budget=request.selected_budget,
        request_sha256=request.request_sha256,
        state=state,
        nuggets=nuggets,
        metadata=metadata,
        backend_attempts=attempts,
        error=error,
    )


def _safe_error(exc: Exception) -> str:
    message = _CONTROL.sub(" ", str(exc)).strip()
    if not message:
        message = type(exc).__name__
    return message[:500]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_safe_cache_root(
    cache_root: Path,
    checker: Callable[[Path], bool] | None,
) -> None:
    artifact_name = "0" * 64 + ".json"
    artifact_probes = (
        cache_root / "raw" / artifact_name,
        cache_root / "validated" / artifact_name,
    )
    if checker is not None:
        try:
            ignored = tuple(checker(path) is True for path in artifact_probes)
        except Exception as exc:
            raise ValueError(
                "could not verify that response cache is Git-ignored"
            ) from exc
        if not all(ignored):
            raise ValueError(
                "cache directory inside a Git worktree must be Git-ignored"
            )
        return
    probe = cache_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    environment = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")
        if key in os.environ
    }
    environment.setdefault("PATH", os.defpath)
    environment.update({"LANG": "C", "LC_ALL": "C"})
    try:
        worktree = subprocess.run(
            ["git", "-C", str(probe), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
            env=environment,
            shell=False,
            text=True,
        )
    except OSError as exc:
        raise ValueError(
            "could not determine whether cache is inside a Git worktree"
        ) from exc
    if worktree.returncode != 0:
        if (
            worktree.returncode == 128
            and "not a git repository" in worktree.stderr.casefold()
        ):
            return
        raise ValueError("could not determine whether cache is inside a Git worktree")
    root_text = worktree.stdout.strip()
    if not root_text:
        raise ValueError("could not determine whether cache is inside a Git worktree")
    worktree_root = Path(root_text).resolve()
    if cache_root != worktree_root and worktree_root not in cache_root.parents:
        raise ValueError("Git worktree identity does not contain the cache directory")
    for artifact_probe in artifact_probes:
        try:
            ignored = subprocess.run(
                [
                    "git",
                    "-C",
                    str(worktree_root),
                    "check-ignore",
                    "-q",
                    "--",
                    str(artifact_probe),
                ],
                capture_output=True,
                check=False,
                env=environment,
                shell=False,
                text=True,
            )
        except OSError as exc:
            raise ValueError(
                "could not verify that response cache is Git-ignored"
            ) from exc
        if ignored.returncode == 1:
            raise ValueError(
                "cache directory inside a Git worktree must be Git-ignored"
            )
        if ignored.returncode != 0:
            raise ValueError("could not verify that response cache is Git-ignored")


def load_validated_selection_artifacts(
    selections_path: Path,
    selection_manifest_path: Path,
) -> tuple[
    tuple[SubnarrativeSelection, ...],
    Mapping[str, object],
    SelectionPolicy,
]:
    """Strictly decode selection rows and their producer manifest."""
    selections_path = Path(selections_path)
    selection_manifest_path = Path(selection_manifest_path)
    selection_bytes = selections_path.read_bytes()
    selections = _load_selections(selection_bytes, selections_path)
    manifest = _decode_json(
        selection_manifest_path.read_bytes(), "selection manifest"
    )
    policy = _validate_selection_manifest(
        manifest, selection_bytes, selections_path, selections
    )
    return selections, manifest, policy


def _load_selections(
    source: bytes,
    path: Path,
) -> tuple[SubnarrativeSelection, ...]:
    selections: list[SubnarrativeSelection] = []
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL rows are not allowed")
        try:
            selection = decode_subnarrative_selection(line)
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: invalid selection: {exc}") from exc
        key = (selection.context.topic_id, selection.context.subnarrative_id)
        if key in seen:
            raise ValueError(
                f"{path}:{line_number}: duplicate topic/subnarrative selection"
            )
        seen.add(key)
        selections.append(selection)
    return tuple(selections)


def _validate_selection_manifest(
    manifest: Mapping[str, object],
    selection_bytes: bytes,
    selection_path: Path,
    selections: tuple[SubnarrativeSelection, ...],
) -> SelectionPolicy:
    _require_fields(
        manifest,
        _SELECTION_MANIFEST_FIELDS,
        "selection manifest",
    )
    expected = {
        "schema_version": SELECTION_MANIFEST_SCHEMA_VERSION,
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "context_schema_version": "subnarrative_selection_context_v1",
        "candidate_schema_version": "extractive_candidate_nugget_v1",
        "records_schema_version": TOPIC_RECORDS_SCHEMA_VERSION,
        "records_stage": CANDIDATE_STAGE,
        "records_file": "records.sqlite3",
        "records_manifest_file": "records-manifest.json",
        "selection_file": selection_path.name,
        "selections_sha256": sha256(selection_bytes).hexdigest(),
        "output_sha256": sha256(selection_bytes).hexdigest(),
        "selection_count": len(selections),
        "context_count": len(selections),
        "retrieval_network_calls": 0,
        "hosted_llm_calls": 0,
    }
    for key, expected_value in expected.items():
        if manifest[key] != expected_value:
            raise ValueError(f"selection manifest {key} identity mismatch")
    for key in (
        "records_database_sha256",
        "candidate_semantic_sha256",
        "contexts_sha256",
        "selections_sha256",
        "output_sha256",
    ):
        if not isinstance(manifest[key], str) or not _SHA256.fullmatch(manifest[key]):
            raise ValueError(
                f"selection manifest {key} must be a lowercase SHA-256 digest"
            )
    for key in (
        "candidate_rows_scanned",
        "candidate_projection_count",
        "loaded_candidate_projection_count",
        "context_count",
        "selection_count",
        "exact_group_count",
        "semantic_cluster_count",
        "selected_cluster_count",
    ):
        if (
            isinstance(manifest[key], bool)
            or not isinstance(manifest[key], int)
            or manifest[key] < 0
        ):
            raise ValueError(
                f"selection manifest {key} must be a non-negative integer"
            )
    for key in ("records_file", "records_manifest_file", "contexts_file"):
        value = manifest[key]
        if (
            not isinstance(value, str)
            or not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or "\x00" in value
        ):
            raise ValueError(
                f"selection manifest {key} must be a non-empty basename"
            )
    scorer = manifest["candidate_scorer_identity"]
    if not isinstance(scorer, Mapping):
        raise ValueError("selection manifest candidate scorer identity must be an object")
    _require_fields(
        scorer,
        _SCORER_IDENTITY_FIELDS,
        "selection manifest candidate scorer identity",
    )
    for key in _SCORER_IDENTITY_FIELDS - {"sentence_max_length"}:
        if not isinstance(scorer[key], str) or not scorer[key]:
            raise ValueError(
                "selection manifest candidate scorer identity values must be non-empty strings"
            )
    if (
        scorer["score_representation"] != "raw_logits"
        or scorer["score_kind"] != "extractive_sentence_v1"
        or scorer["input_policy"] != SCORING_NORMALIZATION_VERSION
    ):
        raise ValueError("selection manifest candidate scorer identity is invalid")
    if (
        isinstance(scorer["sentence_max_length"], bool)
        or not isinstance(scorer["sentence_max_length"], int)
        or scorer["sentence_max_length"] <= 0
    ):
        raise ValueError(
            "selection manifest candidate scorer sentence_max_length must be positive"
        )
    policy_value = manifest["policy"]
    if not isinstance(policy_value, Mapping) or set(policy_value) != {
        "budgets",
        "precluster_limit",
        "semantic_threshold",
        "mmr_lambda",
    }:
        raise ValueError("selection manifest policy is invalid")
    budgets = policy_value["budgets"]
    if not isinstance(budgets, list):
        raise ValueError("selection manifest policy is invalid")
    try:
        policy = SelectionPolicy(
            budgets=tuple(budgets),
            precluster_limit=policy_value["precluster_limit"],
            semantic_threshold=policy_value["semantic_threshold"],
            mmr_lambda=policy_value["mmr_lambda"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("selection manifest policy is invalid") from exc
    expected_policy: dict[str, object] = {
        "budgets": list(policy.budgets),
        "precluster_limit": policy.precluster_limit,
        "semantic_threshold": policy.semantic_threshold,
        "mmr_lambda": policy.mmr_lambda,
    }
    identity = manifest["embedding_identity"]
    if (
        policy_value != expected_policy
        or not isinstance(identity, Mapping)
        or not identity
        or any(not isinstance(key, str) or not key for key in identity)
    ):
        raise ValueError("selection manifest policy or embedding identity mismatch")
    identity_dict = dict(identity)
    if any(
        selection.policy != policy
        or dict(selection.similarity_identity) != identity_dict
        for selection in selections
    ):
        raise ValueError("selections do not share one policy and embedding identity")
    aggregates = {
        "candidate_projection_count": sum(
            selection.candidate_count for selection in selections
        ),
        "exact_group_count": sum(
            selection.exact_group_count for selection in selections
        ),
        "semantic_cluster_count": sum(
            selection.semantic_cluster_count for selection in selections
        ),
        "selected_cluster_count": sum(
            len(selection.clusters) for selection in selections
        ),
    }
    for key, expected_value in aggregates.items():
        if manifest[key] != expected_value:
            raise ValueError(f"selection manifest {key} aggregate mismatch")
    if (
        manifest["loaded_candidate_projection_count"]
        > manifest["candidate_projection_count"]
        or manifest["candidate_rows_scanned"]
        < manifest["candidate_projection_count"]
    ):
        raise ValueError("selection manifest candidate projection counts are inconsistent")
    return policy


def _validate_existing_run(
    output: Path,
    manifest_path: Path,
    expected_identity: Mapping[str, object],
    requests: tuple[CanonicalNuggetRequest, ...],
) -> bool:
    if not manifest_path.exists():
        return False
    if not output.is_file() or not manifest_path.is_file():
        raise ValueError("existing manifest conflicts with requested run")
    try:
        existing = _decode_json(
            manifest_path.read_bytes(),
            "existing canonical nugget manifest",
        )
        _require_fields(
            existing,
            _MANIFEST_FIELDS,
            "existing canonical nugget manifest",
        )
        _validate_existing_manifest_types(existing)
        if any(existing[key] != value for key, value in expected_identity.items()):
            raise ValueError
        output_bytes = output.read_bytes()
        digest = sha256(output_bytes).hexdigest()
        if (
            existing["canonical_nuggets_sha256"] != digest
            or existing["output_sha256"] != digest
            or existing["state_counts"]
            != _existing_output_state_counts(output_bytes, requests)
        ):
            raise ValueError
    except Exception as exc:
        raise ValueError("existing manifest conflicts with requested run") from exc
    return True


def _validate_existing_manifest_types(existing: Mapping[str, object]) -> None:
    text_fields = (
        "schema_version",
        "result_schema_version",
        "canonical_response_schema_version",
        "selection_schema_version",
        "selection_manifest_schema_version",
        "selection_file",
        "selection_manifest_file",
        "canonical_nugget_file",
        "model",
        "prompt_version",
    )
    if any(type(existing[key]) is not str or not existing[key] for key in text_fields):
        raise ValueError("existing canonical nugget manifest text fields are invalid")
    digest_fields = (
        "selections_sha256",
        "selection_manifest_sha256",
        "canonical_nuggets_sha256",
        "output_sha256",
    )
    if any(
        type(existing[key]) is not str or not _SHA256.fullmatch(existing[key])
        for key in digest_fields
    ):
        raise ValueError("existing canonical nugget manifest digests are invalid")
    integer_fields = (
        "selected_budget",
        "max_canonical_claims",
        "max_supporting_documents_per_claim",
        "selection_count",
        "result_count",
        *_RUN_COUNTER_FIELDS,
    )
    if any(type(existing[key]) is not int or existing[key] < 0 for key in integer_fields):
        raise ValueError("existing canonical nugget manifest counts are invalid")
    request_sha256s = existing["request_sha256s"]
    if (
        not isinstance(request_sha256s, list)
        or len(request_sha256s) != existing["result_count"]
        or any(
            type(value) is not str or not _SHA256.fullmatch(value)
            for value in request_sha256s
        )
    ):
        raise ValueError("existing canonical nugget manifest request hashes are invalid")
    state_counts = existing["state_counts"]
    if (
        not isinstance(state_counts, Mapping)
        or not set(state_counts) <= _RESULT_STATES
        or any(
            type(state) is not str or type(count) is not int or count < 0
            for state, count in state_counts.items()
        )
        or sum(state_counts.values()) != existing["result_count"]
    ):
        raise ValueError("existing canonical nugget manifest state counts are invalid")


def _existing_output_state_counts(
    source: bytes,
    requests: tuple[CanonicalNuggetRequest, ...],
) -> dict[str, int]:
    lines = source.splitlines()
    if len(lines) != len(requests) or any(not line.strip() for line in lines):
        raise ValueError("existing canonical output row count is invalid")
    states: Counter[str] = Counter()
    for line, request in zip(lines, requests, strict=True):
        row = _decode_json(line, "existing canonical nugget result")
        states[validate_canonical_nugget_result(row, request)] += 1
    return dict(sorted(states.items()))


class _NoBackend:
    def complete(
        self,
        request: CanonicalNuggetRequest,
    ) -> BackendReply:  # pragma: no cover
        raise AssertionError("empty requests must not call a backend")


class _FixedReplyBackend:
    def __init__(self, reply: BackendReply) -> None:
        self._reply = reply

    def complete(self, request: CanonicalNuggetRequest) -> BackendReply:
        return self._reply


class _RawCachingBackend:
    def __init__(self, delegate: object, path: Path, request_sha256: str) -> None:
        self._delegate = delegate
        self._path = path
        self._request_sha256 = request_sha256
        self.reply: BackendReply | None = None
        self.wrote_raw = False

    def complete(self, request: CanonicalNuggetRequest) -> BackendReply:
        complete = getattr(self._delegate, "complete", None)
        if not callable(complete):
            raise TypeError("backend must provide complete(request)")
        reply = complete(request)
        if not isinstance(reply, BackendReply):
            raise TypeError("backend must return BackendReply")
        response_bodies = _reply_response_bodies(reply)
        metadata_entries = _reply_metadata_entries(reply, len(response_bodies))
        raw = {
            "schema_version": RAW_CACHE_SCHEMA_VERSION,
            "request_sha256": self._request_sha256,
            "status": reply.status,
            "content_base64": base64.b64encode(reply.content).decode("ascii"),
            "response_bodies_base64": [
                base64.b64encode(body).decode("ascii") for body in response_bodies
            ],
            "response_body_sha256s": [
                sha256(body).hexdigest() for body in response_bodies
            ],
            "metadata": dict(reply.metadata),
            "metadata_entries": [dict(entry) for entry in metadata_entries],
        }
        _atomic_write(self._path, _canonical_json(raw) + b"\n")
        self.reply = reply
        self.wrote_raw = True
        return reply


def _reply_response_bodies(reply: BackendReply) -> tuple[bytes, ...]:
    bodies = reply.response_bodies or (reply.response_body,)
    if not bodies or any(not isinstance(body, bytes) for body in bodies):
        raise TypeError("backend reply response bodies must be non-empty bytes")
    if bodies[0] != reply.response_body:
        raise ValueError("backend reply primary response body differs from provenance")
    return tuple(bodies)


def _reply_metadata_entries(
    reply: BackendReply,
    expected_count: int,
) -> tuple[Mapping[str, object], ...]:
    entries = reply.metadata_entries or (reply.metadata,)
    if len(entries) != expected_count or any(
        not isinstance(entry, Mapping) for entry in entries
    ):
        raise TypeError("backend reply metadata provenance is invalid")
    return tuple(entries)


def _transport_invocation_count(backend: object) -> int | None:
    """Read an optional exact hosted-transport counter without constraining backends."""
    try:
        value = getattr(backend, "transport_invocation_count", None)
    except Exception:
        return None
    return value if type(value) is int and value >= 0 else None


def _write_validated(
    path: Path,
    request: CanonicalNuggetRequest,
    reply: BackendReply,
) -> None:
    content_bytes = _canonical_json(
        _decode_json(reply.content, "validated canonical content")
    )
    response_bodies = _reply_response_bodies(reply)
    metadata_entries = _reply_metadata_entries(reply, len(response_bodies))
    cached = {
        "schema_version": VALIDATED_CACHE_SCHEMA_VERSION,
        "request_sha256": request.request_sha256,
        "content": content_bytes.decode("utf-8"),
        "content_sha256": sha256(content_bytes).hexdigest(),
        "status": reply.status,
        "response_body_sha256s": [
            sha256(body).hexdigest() for body in response_bodies
        ],
        "metadata": dict(reply.metadata),
        "metadata_entries": [dict(entry) for entry in metadata_entries],
    }
    _atomic_write(path, _canonical_json(cached) + b"\n")


def _load_raw(
    path: Path,
    request: CanonicalNuggetRequest,
) -> BackendReply | None:
    try:
        source = path.read_bytes()
        cached = _decode_json(source, "raw canonical cache")
        _require_fields(cached, _RAW_CACHE_FIELDS, "raw canonical cache")
        if source != _canonical_json(cached) + b"\n":
            return None
        content_text = cached["content_base64"]
        response_texts = cached["response_bodies_base64"]
        response_hashes = cached["response_body_sha256s"]
        metadata_entries = cached["metadata_entries"]
        if (
            not isinstance(content_text, str)
            or not isinstance(response_texts, list)
            or not response_texts
            or not isinstance(response_hashes, list)
            or len(response_hashes) != len(response_texts)
            or not isinstance(metadata_entries, list)
            or len(metadata_entries) != len(response_texts)
            or any(not isinstance(value, str) for value in response_texts)
            or any(not isinstance(value, Mapping) for value in metadata_entries)
        ):
            return None
        content = base64.b64decode(content_text.encode("ascii"), validate=True)
        response_bodies = tuple(
            base64.b64decode(value.encode("ascii"), validate=True)
            for value in response_texts
        )
        if (
            cached["schema_version"] != RAW_CACHE_SCHEMA_VERSION
            or cached["request_sha256"] != request.request_sha256
            or type(cached["status"]) is not int
            or base64.b64encode(content).decode("ascii") != content_text
            or any(
                base64.b64encode(body).decode("ascii") != encoded
                for body, encoded in zip(response_bodies, response_texts, strict=True)
            )
            or any(
                not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
                or digest != sha256(body).hexdigest()
                for digest, body in zip(response_hashes, response_bodies, strict=True)
            )
            or not isinstance(cached["metadata"], Mapping)
        ):
            return None
        return BackendReply(
            content=content,
            response_body=response_bodies[0],
            status=cached["status"],
            metadata=dict(cached["metadata"]),
            response_bodies=response_bodies,
            metadata_entries=tuple(dict(entry) for entry in metadata_entries),
        )
    except (OSError, ValueError, UnicodeEncodeError, binascii.Error):
        return None


def _load_validated(
    path: Path,
    request: CanonicalNuggetRequest,
) -> BackendReply | None:
    try:
        cached = _decode_json(path.read_bytes(), "validated canonical cache")
        _require_fields(
            cached,
            frozenset(
                {
                    "schema_version",
                    "request_sha256",
                    "content",
                    "content_sha256",
                    "status",
                    "response_body_sha256s",
                    "metadata",
                    "metadata_entries",
                }
            ),
            "validated canonical cache",
        )
        content_bytes = (
            cached["content"].encode("utf-8")
            if isinstance(cached["content"], str)
            else b""
        )
        response_hashes = cached["response_body_sha256s"]
        metadata_entries = cached["metadata_entries"]
        if (
            cached["schema_version"] != VALIDATED_CACHE_SCHEMA_VERSION
            or cached["request_sha256"] != request.request_sha256
            or not isinstance(cached["content"], str)
            or cached["content_sha256"] != sha256(content_bytes).hexdigest()
            or content_bytes
            != _canonical_json(
                _decode_json(content_bytes, "validated canonical content")
            )
            or type(cached["status"]) is not int
            or not isinstance(response_hashes, list)
            or not response_hashes
            or any(
                not isinstance(digest, str) or not _SHA256.fullmatch(digest)
                for digest in response_hashes
            )
            or not isinstance(metadata_entries, list)
            or len(metadata_entries) != len(response_hashes)
            or any(not isinstance(entry, Mapping) for entry in metadata_entries)
            or not isinstance(cached["metadata"], Mapping)
        ):
            return None
        return BackendReply(
            content=content_bytes,
            response_body=b"",
            status=cached["status"],
            metadata=dict(cached["metadata"]),
            metadata_entries=tuple(dict(entry) for entry in metadata_entries),
        )
    except (OSError, ValueError):
        return None


def require_validated_canonical_result(
    cache_dir: Path,
    request: CanonicalNuggetRequest,
) -> CanonicalNuggetResult:
    """Purely read and semantically validate one exact canonical cache hit."""
    if not isinstance(cache_dir, Path):
        raise TypeError("cache_dir must be a Path")
    if not isinstance(request, CanonicalNuggetRequest):
        raise TypeError("request must be CanonicalNuggetRequest")
    if not request.evidence:
        return canonicalize_subnarrative(request, _NoBackend())
    path = cache_dir / "validated" / f"{request.request_sha256}.json"
    if not path.exists():
        raise CanonicalValidatedCacheMiss(
            f"validated canonical cache miss for {request.request_sha256}"
        )
    if not path.is_file():
        raise CanonicalValidatedCacheIntegrityError(
            f"validated canonical cache entry is invalid: {path}"
        )
    try:
        source = path.read_bytes()
        decoded = _decode_json(source, "validated canonical cache")
        if source != _canonical_json(decoded) + b"\n":
            raise ValueError("validated canonical cache is not canonical JSON")
    except (OSError, ValueError) as exc:
        raise CanonicalValidatedCacheIntegrityError(
            f"validated canonical cache entry is invalid: {path}"
        ) from exc
    reply = _load_validated(path, request)
    if reply is None:
        raise CanonicalValidatedCacheIntegrityError(
            f"validated canonical cache entry is invalid: {path}"
        )
    result = canonicalize_subnarrative(request, _FixedReplyBackend(reply))
    if result.state != "complete":
        raise CanonicalValidatedCacheIntegrityError(
            f"validated canonical cache entry is invalid: {path}"
        )
    result = replace(result, backend_attempts=0)
    try:
        validate_canonical_nugget_result(_result_json(result), request)
    except ValueError as exc:
        raise CanonicalValidatedCacheIntegrityError(
            f"validated canonical cache entry is invalid: {path}"
        ) from exc
    return result


def _result_evidence_json(row: CanonicalEvidence) -> dict[str, object]:
    return {
        "candidate_nugget_id": row.candidate_nugget_id,
        "candidate_kind": row.candidate_kind,
        "text": row.text,
        "text_sha256": row.text_sha256,
        "docid": row.docid,
        "document_sha256": row.document_sha256,
        "cluster_id": row.cluster_id,
    }


def _result_json(result: CanonicalNuggetResult) -> dict[str, object]:
    kind = (
        "extractive_fallback"
        if result.state == "fallback_extractive"
        else "model_claim"
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "topic_id": result.topic_id,
        "subnarrative_id": result.subnarrative_id,
        "selected_budget": result.selected_budget,
        "request_sha256": result.request_sha256,
        "state": result.state,
        "nuggets": [
            {
                "canonical_nugget_id": nugget.canonical_nugget_id,
                "nugget_kind": kind,
                "claim_text": nugget.claim_text,
                "importance": nugget.importance,
                "evidence": [
                    _result_evidence_json(row)
                    for row in nugget.evidence
                ],
            }
            for nugget in result.nuggets
        ],
        "metadata": dict(result.metadata),
        "error": result.error,
    }


def _require_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has unexpected or missing fields")


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
