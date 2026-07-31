"""One-topic probe for centralized post-batch canonical nuggetization.

Question: can one bounded Nuggetizer call canonicalize the provisional claims
already returned by the topic-224 researchers while preserving their exact
snippet provenance? This runner reconstructs one existing Phoenix trace in
memory, compares the accepted ledger with the centralized output, and never
writes trace or model content to disk.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
import sys
from time import monotonic
from typing import Any, Literal

from phoenix.client import Client

from trec_rag.canonical_nuggets import (
    MAX_CANONICAL_NUGGETS,
    MAX_SUPPORTING_DOCUMENTS_PER_CLAIM,
    CanonicalEvidence,
    CanonicalNuggetRequest,
    CanonicalNuggetResult,
    canonicalize_subnarrative,
)
from trec_rag.deepagent_research import EvidenceBundle
from trec_rag.facet_extraction import OPENROUTER_DEEPSEEK_MODEL
from trec_rag.nuggetizer_adapter import (
    NuggetizerCanonicalNuggetBackend,
    render_nuggetizer_request_body,
)


DEFAULT_TRACE_ID = "f9d00a71e02ac6f940deb75a5090db5b"
DEFAULT_TOPIC_ID = "224"
EXPECTED_BUNDLES = 5
EXPECTED_BASELINE_NUGGETS = 6
EXPECTED_SNIPPETS = 16
InputSource = Literal["researcher", "ledger"]


@dataclass(frozen=True)
class SnippetObservation:
    document_id: str
    snippet_id: str
    page_index: int
    text: str


@dataclass(frozen=True)
class GroundedClaim:
    claim_id: str
    text: str
    evidence: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class TraceSnapshot:
    narrative: str
    bundles: tuple[EvidenceBundle, ...]
    snippets: Mapping[str, SnippetObservation]
    baseline: tuple[GroundedClaim, ...]


@dataclass(frozen=True)
class PreparedProbe:
    input_source: InputSource
    snapshot: TraceSnapshot
    provisional: tuple[GroundedClaim, ...]
    request: CanonicalNuggetRequest
    aliases_by_provisional: Mapping[str, tuple[str, ...]]
    grounding_failures: tuple[str, ...]


def _json_object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} is missing JSON text")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to an object")
    return parsed


def _tool_message_content(value: object, *, label: str) -> str:
    outer = _json_object(value, label=label)
    try:
        content = outer["data"]["content"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{label} lacks tool-message content") from exc
    if not isinstance(content, str):
        raise ValueError(f"{label} tool-message content is not text")
    return content


def _task_bundle(value: object) -> EvidenceBundle:
    outer = _json_object(value, label="task output")
    try:
        messages = outer["update"]["messages"]
        content = messages[0]["data"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("task output lacks the structured researcher message") from exc
    if not isinstance(messages, list) or len(messages) != 1 or not isinstance(content, str):
        raise ValueError("task output must contain exactly one structured message")
    return EvidenceBundle.model_validate_json(content)


def _sequence_attribute(
    attributes: Mapping[str, object], key: str, *, expected: int | None = None
) -> list[object]:
    value = attributes.get(key)
    if not isinstance(value, list):
        raise ValueError(f"trace attribute {key!r} is not a list")
    if expected is not None and len(value) != expected:
        raise ValueError(f"trace attribute {key!r} has inconsistent length")
    return value


def reconstruct_trace(spans: Sequence[Mapping[str, object]]) -> TraceSnapshot:
    """Purely reconstruct researcher, snippet, and accepted-ledger state."""
    root = [span for span in spans if span.get("name") == "deepagent.retrieve"]
    if len(root) != 1:
        raise ValueError("trace must contain exactly one deepagent.retrieve span")
    root_attributes = root[0].get("attributes")
    if not isinstance(root_attributes, Mapping):
        raise ValueError("root span lacks attributes")
    narrative = root_attributes.get("input.value")
    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError("root span lacks the untouched narrative")
    narrative = narrative.strip()

    task_spans = sorted(
        (span for span in spans if span.get("name") == "task"),
        key=lambda span: str(span.get("start_time", "")),
    )
    bundles: list[EvidenceBundle] = []
    for span in task_spans:
        attributes = span.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError("task span lacks attributes")
        bundle = _task_bundle(attributes.get("output.value"))
        traced_id = attributes.get("metadata.deepagent.research_task_id")
        if traced_id != bundle.research_task_id:
            raise ValueError("task span identity does not match its EvidenceBundle")
        bundles.append(bundle)

    snippets: dict[str, SnippetObservation] = {}
    for span in sorted(
        (item for item in spans if item.get("name") == "deepagent.extract_relevant_snippets"),
        key=lambda item: str(item.get("start_time", "")),
    ):
        attributes = span.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError("snippet span lacks attributes")
        document_id = attributes.get("snippet.document_id")
        page_index = attributes.get("snippet.page_index")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("snippet span lacks document identity")
        if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
            raise ValueError("snippet span has invalid page index")
        ids = _sequence_attribute(attributes, "snippet.chunk_ids")
        texts = _sequence_attribute(attributes, "snippet.texts", expected=len(ids))
        for snippet_id, text in zip(ids, texts, strict=True):
            if not isinstance(snippet_id, str) or not snippet_id:
                raise ValueError("snippet span contains an invalid snippet ID")
            if not isinstance(text, str) or not text:
                raise ValueError("snippet span contains empty snippet text")
            observation = SnippetObservation(document_id, snippet_id, page_index, text)
            if snippet_id in snippets and snippets[snippet_id] != observation:
                raise ValueError(f"snippet {snippet_id!r} has conflicting observations")
            snippets[snippet_id] = observation

    accepted: dict[str, GroundedClaim] = {}
    update_spans = sorted(
        (span for span in spans if span.get("name") == "update_retrieval_state"),
        key=lambda span: str(span.get("start_time", "")),
    )
    for span in update_spans:
        attributes = span.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError("state-update span lacks attributes")
        request = _json_object(attributes.get("input.value"), label="state-update input")
        result = _json_object(
            _tool_message_content(attributes.get("output.value"), label="state-update output"),
            label="state-update result",
        )
        delta = request.get("delta")
        accepted_ids = result.get("accepted_ids")
        if not isinstance(delta, Mapping) or not isinstance(accepted_ids, list):
            raise ValueError("state-update span has an invalid delta/result pair")
        rows = delta.get("add_nuggets", [])
        if not isinstance(rows, list):
            raise ValueError("add_nuggets must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("add_nuggets contains a non-object")
            nugget_id = row.get("nugget_id")
            text = row.get("text")
            evidence = row.get("evidence")
            if nugget_id not in accepted_ids:
                continue
            if (
                not isinstance(nugget_id, str)
                or not isinstance(text, str)
                or not text
                or not isinstance(evidence, list)
                or not evidence
                or any(not isinstance(item, Mapping) for item in evidence)
            ):
                raise ValueError("accepted nugget has an invalid structure")
            accepted[nugget_id] = GroundedClaim(
                nugget_id, text, tuple(evidence)
            )

    baseline = tuple(accepted[key] for key in sorted(accepted))
    return TraceSnapshot(narrative, tuple(bundles), snippets, baseline)


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def grounding_failures(
    claims: Sequence[GroundedClaim], snippets: Mapping[str, SnippetObservation]
) -> tuple[str, ...]:
    failures: list[str] = []
    for claim in claims:
        for index, evidence in enumerate(claim.evidence, start=1):
            snippet_id = evidence.get("snippet_id")
            document_id = evidence.get("document_id")
            page_index = evidence.get("page_index")
            quote = evidence.get("quote")
            prefix = f"{claim.claim_id}:e{index}"
            if not isinstance(snippet_id, str) or snippet_id not in snippets:
                failures.append(f"{prefix}:unknown_snippet")
                continue
            observed = snippets[snippet_id]
            if document_id is not None and document_id != observed.document_id:
                failures.append(f"{prefix}:document_mismatch")
            if page_index is not None and page_index != observed.page_index:
                failures.append(f"{prefix}:page_mismatch")
            if not isinstance(quote, str) or _normalized(quote) not in _normalized(observed.text):
                failures.append(f"{prefix}:ungrounded_quote")
    return tuple(failures)


def prepare_probe(
    snapshot: TraceSnapshot,
    *,
    topic_id: str,
    input_source: InputSource = "researcher",
) -> PreparedProbe:
    provisional: list[GroundedClaim] = []
    if input_source == "researcher":
        for bundle in snapshot.bundles:
            for index, candidate in enumerate(bundle.candidate_nuggets, start=1):
                provisional.append(
                    GroundedClaim(
                        f"{bundle.research_task_id}:p{index:03d}",
                        candidate.claim,
                        tuple(item.model_dump() for item in candidate.evidence),
                    )
                )
    elif input_source == "ledger":
        provisional.extend(snapshot.baseline)
    else:
        raise ValueError(f"unsupported input source: {input_source!r}")

    failures = (
        *grounding_failures(snapshot.baseline, snapshot.snippets),
        *grounding_failures(provisional, snapshot.snippets),
    )

    canonical_evidence: list[CanonicalEvidence] = []
    fallbacks: list[CanonicalEvidence] = []
    aliases_by_provisional: dict[str, tuple[str, ...]] = {}
    alias_index = 1
    for claim in provisional:
        claim_aliases: list[str] = []
        for evidence_index, item in enumerate(claim.evidence, start=1):
            snippet_id = item.get("snippet_id")
            if not isinstance(snippet_id, str) or snippet_id not in snapshot.snippets:
                continue
            observed = snapshot.snippets[snippet_id]
            alias = f"e{alias_index:03d}"
            alias_index += 1
            row = CanonicalEvidence(
                alias=alias,
                cluster_alias=claim.claim_id,
                cluster_id=claim.claim_id,
                candidate_nugget_id=f"{claim.claim_id}:e{evidence_index}",
                candidate_kind="researcher_claim",
                text=claim.text,
                text_sha256=sha256(claim.text.encode("utf-8")).hexdigest(),
                docid=observed.document_id,
                document_sha256=sha256(observed.text.encode("utf-8")).hexdigest(),
            )
            canonical_evidence.append(row)
            claim_aliases.append(alias)
        aliases_by_provisional[claim.claim_id] = tuple(claim_aliases)
        if claim_aliases:
            fallbacks.append(canonical_evidence[-len(claim_aliases)])

    evidence_tuple = tuple(canonical_evidence)
    body = render_nuggetizer_request_body(
        topic_id=topic_id,
        subnarrative_id=f"topic-{topic_id}-post-batch",
        subnarrative_text=snapshot.narrative,
        evidence=evidence_tuple,
        max_canonical_claims=MAX_CANONICAL_NUGGETS,
        max_supporting_documents_per_claim=MAX_SUPPORTING_DOCUMENTS_PER_CLAIM,
    )
    request = CanonicalNuggetRequest(
        topic_id=topic_id,
        subnarrative_id=f"topic-{topic_id}-post-batch",
        subnarrative_text=snapshot.narrative,
        selected_budget=len(provisional),
        max_canonical_claims=MAX_CANONICAL_NUGGETS,
        max_supporting_documents_per_claim=MAX_SUPPORTING_DOCUMENTS_PER_CLAIM,
        evidence=evidence_tuple,
        fallback_evidence=tuple(fallbacks),
        request_body=body,
        request_sha256=sha256(body).hexdigest(),
    )
    return PreparedProbe(
        input_source,
        snapshot,
        tuple(provisional),
        request,
        aliases_by_provisional,
        tuple(failures),
    )


def assert_preflight(prepared: PreparedProbe) -> None:
    if len(prepared.snapshot.bundles) != EXPECTED_BUNDLES:
        raise ValueError(
            f"expected {EXPECTED_BUNDLES} bundles, got {len(prepared.snapshot.bundles)}"
        )
    if len(prepared.snapshot.baseline) != EXPECTED_BASELINE_NUGGETS:
        raise ValueError(
            "expected "
            f"{EXPECTED_BASELINE_NUGGETS} baseline nuggets, "
            f"got {len(prepared.snapshot.baseline)}"
        )
    if len(prepared.provisional) != EXPECTED_BASELINE_NUGGETS:
        raise ValueError(
            "expected "
            f"{EXPECTED_BASELINE_NUGGETS} provisional nuggets, "
            f"got {len(prepared.provisional)}"
        )
    if len(prepared.snapshot.snippets) != EXPECTED_SNIPPETS:
        raise ValueError(
            f"expected {EXPECTED_SNIPPETS} snippets, got {len(prepared.snapshot.snippets)}"
        )
    if prepared.grounding_failures:
        raise ValueError(
            f"preflight found {len(prepared.grounding_failures)} grounding failures"
        )
    if len(prepared.request.evidence) != len(prepared.provisional):
        raise ValueError("each provisional nugget must have exactly one evidence alias")


def _exact_duplicate_groups(claims: Sequence[GroundedClaim]) -> list[list[str]]:
    grouped: dict[str, list[str]] = {}
    for claim in claims:
        grouped.setdefault(_normalized(claim.text), []).append(claim.claim_id)
    return [ids for ids in grouped.values() if len(ids) > 1]


def comparison(
    prepared: PreparedProbe,
    result: CanonicalNuggetResult | None,
    *,
    trace_id: str,
    topic_id: str,
    hosted_calls: int,
    latency_seconds: float,
) -> dict[str, object]:
    preflight_passed = (
        len(prepared.snapshot.bundles) == EXPECTED_BUNDLES
        and len(prepared.snapshot.baseline) == EXPECTED_BASELINE_NUGGETS
        and len(prepared.provisional) == EXPECTED_BASELINE_NUGGETS
        and len(prepared.snapshot.snippets) == EXPECTED_SNIPPETS
        and not prepared.grounding_failures
        and len(prepared.request.evidence) == len(prepared.provisional)
    )
    base: dict[str, object] = {
        "topic_id": topic_id,
        "trace_id": trace_id,
        "model": OPENROUTER_DEEPSEEK_MODEL,
        "input_source": prepared.input_source,
        "preflight": {
            "passed": preflight_passed,
            "researcher_bundles": len(prepared.snapshot.bundles),
            "productive_bundles": sum(bool(bundle.candidate_nuggets) for bundle in prepared.snapshot.bundles),
            "observed_snippets": len(prepared.snapshot.snippets),
            "baseline_nuggets": len(prepared.snapshot.baseline),
            "provisional_nuggets": len(prepared.provisional),
            "grounding_failures": list(prepared.grounding_failures),
            "exact_normalized_duplicate_groups": _exact_duplicate_groups(prepared.provisional),
        },
        "path_a": [
            {
                "nugget_id": claim.claim_id,
                "claim": claim.text,
                "documents": sorted(
                    {
                        prepared.snapshot.snippets[str(item.get("snippet_id"))].document_id
                        for item in claim.evidence
                        if isinstance(item.get("snippet_id"), str)
                        and item.get("snippet_id") in prepared.snapshot.snippets
                    }
                ),
            }
            for claim in prepared.snapshot.baseline
        ],
        "hosted_calls": hosted_calls,
        "latency_seconds": round(latency_seconds, 3),
    }
    if result is None:
        base["mode"] = "dry_run"
        base["path_b"] = None
        return base

    canonical_aliases = {
        nugget.canonical_nugget_id: tuple(row.alias for row in nugget.evidence)
        for nugget in result.nuggets
    }
    alias_to_canonical = {
        alias: canonical_id
        for canonical_id, aliases in canonical_aliases.items()
        for alias in aliases
    }
    mappings = []
    for provisional in prepared.provisional:
        aliases = prepared.aliases_by_provisional[provisional.claim_id]
        mapped = sorted(
            {alias_to_canonical[alias] for alias in aliases if alias in alias_to_canonical}
        )
        mappings.append(
            {
                "provisional_id": provisional.claim_id,
                "claim": provisional.text,
                "evidence_aliases": list(aliases),
                "canonical_ids": mapped,
            }
        )
    all_aliases = {row.alias for row in prepared.request.evidence}
    retained = set(alias_to_canonical)
    mapped_counts = Counter(
        canonical_id
        for item in mappings
        for canonical_id in item["canonical_ids"]  # type: ignore[index]
    )
    base.update(
        {
            "mode": "hosted",
            "state": result.state,
            "error": result.error,
            "path_b": [
                {
                    "canonical_id": nugget.canonical_nugget_id,
                    "claim": nugget.claim_text,
                    "evidence_aliases": list(canonical_aliases[nugget.canonical_nugget_id]),
                    "documents": sorted({row.docid for row in nugget.evidence}),
                }
                for nugget in result.nuggets
            ],
            "provisional_to_canonical": mappings,
            "candidate_paraphrase_merges": [
                canonical_id
                for canonical_id, count in mapped_counts.items()
                if count > 1
            ],
            "retained_evidence_aliases": sorted(retained),
            "orphaned_evidence_aliases": sorted(all_aliases - retained),
            "unknown_evidence_aliases": sorted(retained - all_aliases),
            "output_exact_duplicate_claims": len(
                result.nuggets
            ) - len({_normalized(nugget.claim_text) for nugget in result.nuggets}),
            "provider": result.metadata.get("provider"),
            "usage": result.metadata.get("usage", {}),
        }
    )
    return base


def _client_from_env() -> tuple[Client, str]:
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip().rstrip("/")
    api_key = os.environ.get("PHOENIX_API_KEY", "").strip()
    project = os.environ.get("PHOENIX_PROJECT_NAME", "").strip()
    if not endpoint or not api_key or not project:
        raise ValueError(
            "PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_API_KEY, and "
            "PHOENIX_PROJECT_NAME must be set"
        )
    return Client(base_url=endpoint, api_key=api_key), project


def run(
    *,
    trace_id: str,
    topic_id: str,
    input_source: InputSource,
    dry_run: bool,
) -> dict[str, object]:
    client, project = _client_from_env()
    spans = client.spans.get_spans(
        project_identifier=project,
        trace_ids=[trace_id],
        limit=1_000,
        timeout=20,
    )
    snapshot = reconstruct_trace(spans)
    prepared = prepare_probe(
        snapshot,
        topic_id=topic_id,
        input_source=input_source,
    )
    if dry_run:
        return comparison(
            prepared,
            None,
            trace_id=trace_id,
            topic_id=topic_id,
            hosted_calls=0,
            latency_seconds=0.0,
        )

    assert_preflight(prepared)
    backend = NuggetizerCanonicalNuggetBackend()
    started = monotonic()
    result = canonicalize_subnarrative(prepared.request, backend)
    elapsed = monotonic() - started
    return comparison(
        prepared,
        result,
        trace_id=trace_id,
        topic_id=topic_id,
        hosted_calls=backend.transport_invocation_count,
        latency_seconds=elapsed,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-id", default=DEFAULT_TRACE_ID)
    parser.add_argument("--topic-id", default=DEFAULT_TOPIC_ID)
    parser.add_argument(
        "--input-source",
        choices=("researcher", "ledger"),
        default="researcher",
        help="canonicalize raw researcher claims or mechanically grounded ledger nuggets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="reconstruct and validate the trace without calling OpenRouter",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = run(
            trace_id=arguments.trace_id,
            topic_id=arguments.topic_id,
            input_source=arguments.input_source,
            dry_run=arguments.dry_run,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
