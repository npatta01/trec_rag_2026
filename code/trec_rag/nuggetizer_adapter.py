"""Bounded bridge from the canonical nugget contract to Nuggetizer 0.0.5."""

from __future__ import annotations

import ast
from dataclasses import replace
from hashlib import sha256
import json
import logging
from typing import TYPE_CHECKING, Mapping

from trec_rag.facet_extraction import (
    BackendReply,
    MAX_COMPLETION_TOKENS,
    OPENROUTER_DEEPSEEK_MODEL,
)

if TYPE_CHECKING:
    from trec_rag.canonical_nuggets import CanonicalEvidence, CanonicalNuggetRequest


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _upstream_request(
    *,
    subnarrative_id: str,
    subnarrative_text: str,
    evidence: tuple[CanonicalEvidence, ...],
) -> object:
    """Construct the only upstream request, preserving the sealed join key."""
    from nuggetizer.core.types import Document, Query, Request

    return Request(
        query=Query(qid=subnarrative_id, text=subnarrative_text),
        documents=[
            Document(
                docid=row.candidate_nugget_id,
                segment=f"{row.alias}: {row.text}",
            )
            for row in evidence
        ],
    )


def _creator_messages(
    package_request: object,
    *,
    max_canonical_claims: int,
) -> list[dict[str, str]]:
    """Use the installed package's renderer without its eager network constructor."""
    from nuggetizer.models.nuggetizer import Nuggetizer

    package = object.__new__(Nuggetizer)
    package.creator_max_nuggets = max_canonical_claims
    return package._create_nugget_prompt(  # type: ignore[no-any-return]
        package_request, 0, len(package_request.documents), []
    )


def _response_schema(
    aliases: tuple[str, ...], *, max_canonical_claims: int
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
                            "maxLength": 1000,
                        },
                        "evidence_aliases": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {"type": "string", "enum": list(aliases)},
                        },
                    },
                },
            },
        },
    }


def _wire_body(
    messages: list[dict[str, str]],
    *,
    topic_id: str,
    subnarrative_id: str,
    evidence: tuple[CanonicalEvidence, ...],
    max_canonical_claims: int,
    max_supporting_documents_per_claim: int,
) -> bytes:
    aliases = tuple(row.alias for row in evidence)
    grounded_contract = {
        "topic_id": topic_id,
        "subnarrative_id": subnarrative_id,
        "evidence_aliases": list(aliases),
        "output": {
            "claims": "Return only grounded claims and their supplied aliases.",
            "max_claims": max_canonical_claims,
            "max_supporting_documents_per_claim": max_supporting_documents_per_claim,
        },
    }
    payload = {
        "model": OPENROUTER_DEEPSEEK_MODEL,
        "messages": [*messages, {"role": "user", "content": _canonical_json(grounded_contract).decode("utf-8")}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "canonical_nuggets_v1",
                "strict": True,
                "schema": _response_schema(
                    aliases, max_canonical_claims=max_canonical_claims
                ),
            },
        },
        "provider": {"require_parameters": True, "data_collection": "deny"},
        "reasoning": {"enabled": False},
        "temperature": 0,
        "seed": 0,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
    }
    return _canonical_json(payload)


def render_nuggetizer_request_body(
    *,
    topic_id: str,
    subnarrative_id: str,
    subnarrative_text: str,
    evidence: tuple[CanonicalEvidence, ...],
    max_canonical_claims: int,
    max_supporting_documents_per_claim: int,
) -> bytes:
    """Render the exact package-adapted OpenRouter request used at runtime."""
    package_request = _upstream_request(
        subnarrative_id=subnarrative_id,
        subnarrative_text=subnarrative_text,
        evidence=evidence,
    )
    return _wire_body(
        _creator_messages(
            package_request, max_canonical_claims=max_canonical_claims
        ),
        topic_id=topic_id,
        subnarrative_id=subnarrative_id,
        evidence=evidence,
        max_canonical_claims=max_canonical_claims,
        max_supporting_documents_per_claim=max_supporting_documents_per_claim,
    )


class _CreatorHandler:
    def __init__(self, request: CanonicalNuggetRequest, backend: object) -> None:
        self.request = request
        self.backend = backend
        self.reply: BackendReply | None = None
        self.claims: tuple[object, ...] | None = None
        self.error: Exception | None = None

    def run(
        self, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> tuple[str, int]:
        try:
            body = _wire_body(
                messages,
                topic_id=self.request.topic_id,
                subnarrative_id=self.request.subnarrative_id,
                evidence=self.request.evidence,
                max_canonical_claims=self.request.max_canonical_claims,
                max_supporting_documents_per_claim=(
                    self.request.max_supporting_documents_per_claim
                ),
            )
            if body != self.request.request_body:
                raise ValueError("Nuggetizer creator rendering differs from the sealed request")
            reply = self.backend.complete(
                replace(
                    self.request,
                    request_body=body,
                    request_sha256=sha256(body).hexdigest(),
                )
            )
            from trec_rag.canonical_nuggets import _parse_claims

            claims = _parse_claims(reply.content, self.request)
            self.reply = reply
            self.claims = claims
            return repr([claim.claim_text for claim in claims]), 0
        except Exception as exc:
            self.error = exc
            # A parseable empty list prevents upstream's unbounded parse retry.
            return "[]", 0


class _LocalScorerHandler:
    def run(
        self, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> tuple[str, int]:
        del temperature
        if not messages:
            raise ValueError("Nuggetizer scorer prompt is missing")
        user_content = messages[-1]["content"]
        marker = "Nugget List: "
        try:
            list_text = user_content.split(marker, 1)[1].split("\n", 1)[0]
            count = len(ast.literal_eval(list_text))
        except Exception as exc:
            raise ValueError("Nuggetizer scorer prompt is malformed") from exc
        return repr(["okay"] * count), 0


class NuggetizerCanonicalNuggetBackend:
    """One-call canonical backend using Nuggetizer's public ``create`` method."""

    one_shot_no_retry = True
    redirects_allowed = False

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        transport: object | None = None,
    ) -> None:
        from trec_rag.canonical_nuggets import OpenRouterCanonicalNuggetBackend

        self._openrouter = OpenRouterCanonicalNuggetBackend(
            environ=environ, transport=transport
        )

    def complete(self, request: CanonicalNuggetRequest) -> BackendReply:
        from nuggetizer.models.nuggetizer import Nuggetizer

        package_request = _upstream_request(
            subnarrative_id=request.subnarrative_id,
            subnarrative_text=request.subnarrative_text,
            evidence=request.evidence,
        )
        creator = _CreatorHandler(request, self._openrouter)
        package = object.__new__(Nuggetizer)
        package.creator_window_size = len(package_request.documents)
        package.scorer_window_size = request.max_canonical_claims
        package.creator_max_nuggets = request.max_canonical_claims
        package.scorer_max_nuggets = request.max_canonical_claims
        package.creator_llm = creator
        package.scorer_llm = _LocalScorerHandler()
        package.log_level = 0
        package.logger = logging.getLogger(__name__)
        scored_nuggets = package.create(package_request)
        if creator.error is not None:
            raise creator.error
        if creator.reply is None or creator.claims is None:
            raise RuntimeError("Nuggetizer completed without a creator reply")
        claims_by_text = {claim.claim_text: claim for claim in creator.claims}
        if len(claims_by_text) != len(creator.claims):
            raise ValueError("Nuggetizer creator claims are not uniquely translatable")
        translated_claims = []
        if not isinstance(scored_nuggets, list) or len(scored_nuggets) != len(creator.claims):
            raise ValueError("Nuggetizer returned an incomplete canonical claim set")
        for nugget in scored_nuggets:
            text = getattr(nugget, "text", None)
            if not isinstance(text, str):
                raise ValueError("Nuggetizer returned a malformed canonical claim")
            claim = claims_by_text.pop(text, None)
            if claim is None:
                raise ValueError("Nuggetizer returned an unknown canonical claim")
            translated_claims.append(
                {
                    "claim": claim.claim_text,
                    "evidence_aliases": [row.alias for row in claim.evidence],
                }
            )
        if claims_by_text:
            raise ValueError("Nuggetizer omitted a canonical claim")
        return BackendReply(
            content=_canonical_json({"claims": translated_claims}),
            response_body=creator.reply.response_body,
            status=creator.reply.status,
            metadata=creator.reply.metadata,
        )
