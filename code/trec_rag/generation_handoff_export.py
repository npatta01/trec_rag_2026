"""Project sealed retrieval selections into the generation handoff contract.

This module is the only adapter between retrieval's private ``TopicRecords``
representation and the self-contained selected-evidence manifest consumed by
generation.  It reconstructs only candidates named by the selected snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from types import MappingProxyType

from trec_rag.canonical_nuggets import (
    build_canonical_nugget_request,
    validate_canonical_nugget_result,
)
from trec_rag.facet_evidence import (
    EvidenceMember,
    ExtractiveCandidate,
    SemanticCluster,
    SubnarrativeSelection,
)
from trec_rag.generation_handoff import (
    ClaimHint,
    EvidenceGroup,
    EvidencePassage,
    EvidenceSourceSpan,
    GenerationHandoff,
    GenerationTopic,
    HandoffProducer,
    SOURCE_CONTRACT,
    SelectedCluster,
    TopicSourceReceipts,
    serialize_generation_handoff,
)
from trec_rag.topics import Topic


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PreparedGenerationHandoffArtifact:
    """Complete root handoff bytes awaiting manifest-last publication."""

    path: Path
    body: bytes
    sha256: str
    handoff: GenerationHandoff


@dataclass(frozen=True)
class ValidatedGenerationSnapshot:
    """Evidence projection inputs already bound to sealed retrieval artifacts.

    The retrieval checkpoint loader constructs this only after cross-checking
    the official topic, selected-document ranks, selection manifest,
    TopicRecords stage seal, configured snapshot, fallback identity, and
    canonical-result manifest.  The projector accepts no artifact paths and
    cannot widen the selected candidate set.
    """

    topic: Topic
    official_topics_sha256: str
    retrieval_topic_sha256: str
    selections: tuple[SubnarrativeSelection, ...]
    candidates: Mapping[tuple[str, str], ExtractiveCandidate]
    canonical_results: tuple[Mapping[str, object], ...]
    selected_budget: int
    max_canonical_claims: int
    max_supporting_documents_per_claim: int
    document_ranks: Mapping[str, int]
    original_narrative_fallback: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.topic, Topic):
            raise TypeError("topic must be Topic")
        for value, label in (
            (self.official_topics_sha256, "official_topics_sha256"),
            (self.retrieval_topic_sha256, "retrieval_topic_sha256"),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.selections, tuple)
            or not self.selections
            or any(
                not isinstance(selection, SubnarrativeSelection)
                for selection in self.selections
            )
        ):
            raise ValueError("selections must be a non-empty validated tuple")
        if not isinstance(self.candidates, Mapping) or any(
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(part, str) or not part for part in key)
            or not isinstance(candidate, ExtractiveCandidate)
            for key, candidate in self.candidates.items()
        ):
            raise ValueError("candidates must map typed selection keys to candidates")
        if not isinstance(self.canonical_results, tuple) or any(
            not isinstance(result, Mapping) for result in self.canonical_results
        ):
            raise ValueError("canonical_results must be a tuple of result mappings")
        for value, label in (
            (self.selected_budget, "selected_budget"),
            (self.max_canonical_claims, "max_canonical_claims"),
            (
                self.max_supporting_documents_per_claim,
                "max_supporting_documents_per_claim",
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if not isinstance(self.document_ranks, Mapping):
            raise TypeError("document_ranks must be a mapping")
        if not isinstance(self.original_narrative_fallback, bool):
            raise TypeError("original_narrative_fallback must be boolean")
        object.__setattr__(
            self, "candidates", MappingProxyType(dict(self.candidates))
        )
        object.__setattr__(
            self, "document_ranks", MappingProxyType(dict(self.document_ranks))
        )


def prepare_generation_handoff_artifact(
    *,
    output_dir: Path,
    retrieval_run_id: str,
    producer_revision: str,
    topics: Sequence[GenerationTopic],
) -> PreparedGenerationHandoffArtifact:
    """Assemble and authenticate a handoff without mutating export state.

    Retrieval export owns publication so it can write organizer artifacts and
    this self-contained manifest before sealing their shared outer receipt.
    """
    if isinstance(topics, (str, bytes)) or not isinstance(topics, Sequence):
        raise TypeError("topics must be a sequence of GenerationTopic")
    handoff = GenerationHandoff(
        producer=HandoffProducer(
            source_contract=SOURCE_CONTRACT,
            retrieval_run_id=retrieval_run_id,
            producer_revision=producer_revision,
        ),
        topics=tuple(topics),
    )
    body = serialize_generation_handoff(handoff)
    return PreparedGenerationHandoffArtifact(
        path=Path(output_dir) / "generation_handoff_manifest.json",
        body=body,
        sha256=sha256(body).hexdigest(),
        handoff=handoff,
    )


def _document_ranks(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("document_ranks must be a non-empty mapping")
    result: dict[str, int] = {}
    rank_to_docid: dict[int, str] = {}
    for docid, rank in value.items():
        if not isinstance(docid, str) or not docid:
            raise ValueError("document_ranks keys must be non-empty docids")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("document ranks must be positive integers")
        previous = rank_to_docid.setdefault(rank, docid)
        if previous != docid:
            raise ValueError("one retrieval rank identifies multiple documents")
        result[docid] = rank
    return result


def _selected_clusters(
    selection: SubnarrativeSelection,
    selected_budget: int,
) -> tuple[SemanticCluster, ...]:
    snapshots = {snapshot.budget: snapshot for snapshot in selection.snapshots}
    snapshot = snapshots.get(selected_budget)
    if snapshot is None:
        raise ValueError(
            f"selection {selection.context.subnarrative_id!r} lacks selected budget"
        )
    by_id = {cluster.cluster_id: cluster for cluster in selection.clusters}
    try:
        return tuple(by_id[cluster_id] for cluster_id in snapshot.cluster_ids)
    except KeyError as exc:  # pragma: no cover - guarded by selection validation
        raise ValueError("selected snapshot names an unknown cluster") from exc


def _candidate_matches_member(
    candidate: ExtractiveCandidate,
    member: EvidenceMember,
    *,
    subnarrative_id: str,
    subnarrative_sha256: str,
) -> bool:
    return (
        candidate.subnarrative_id == subnarrative_id
        and candidate.subnarrative_sha256 == subnarrative_sha256
        and candidate.candidate_nugget_id == member.candidate_nugget_id
        and candidate.candidate_kind == member.candidate_kind
        and candidate.text == member.text
        and candidate.docid == member.docid
        and candidate.document_sha256 == member.document_sha256
        and candidate.sentence_cross_encoder_score == member.raw_logit
    )


def _source_span(candidate: ExtractiveCandidate) -> EvidenceSourceSpan:
    if not candidate.evidence_sentences:
        raise ValueError("selected candidate has no exact evidence sentence")
    first = candidate.evidence_sentences[0]
    last = candidate.evidence_sentences[-1]
    span = EvidenceSourceSpan(
        start_char=first.start_char,
        end_char=last.end_char,
        start_byte=first.start_byte,
        end_byte=last.end_byte,
    )
    if (
        span.end_char - span.start_char != len(candidate.text)
        or span.end_byte - span.start_byte != len(candidate.text.encode("utf-8"))
    ):
        raise ValueError("selected candidate offsets do not span its exact UTF-8 text")
    return span


def project_generation_topic(
    snapshot: ValidatedGenerationSnapshot,
) -> GenerationTopic:
    """Project one sealed topic snapshot into selected generation evidence.

    Complete document text and artifact paths are absent from ``snapshot`` and
    therefore cannot be copied into the returned handoff record.
    """
    if not isinstance(snapshot, ValidatedGenerationSnapshot):
        raise TypeError("snapshot must be ValidatedGenerationSnapshot")
    topic = snapshot.topic
    selections = snapshot.selections
    canonical_results = snapshot.canonical_results
    selected_budget = snapshot.selected_budget
    max_canonical_claims = snapshot.max_canonical_claims
    max_supporting_documents_per_claim = (
        snapshot.max_supporting_documents_per_claim
    )
    original_narrative_fallback = snapshot.original_narrative_fallback
    ranks = _document_ranks(snapshot.document_ranks)

    group_ids: set[str] = set()
    selected_by_group: list[tuple[SubnarrativeSelection, tuple[SemanticCluster, ...]]] = []
    required_keys: set[tuple[str, str]] = set()
    for selection in selections:
        context = selection.context
        if (
            context.topic_id != topic.id
            or context.official_narrative != topic.narrative
            or context.subnarrative_id in group_ids
        ):
            raise ValueError("selection context differs from the official topic")
        group_ids.add(context.subnarrative_id)
        clusters = _selected_clusters(selection, selected_budget)
        if not clusters:
            raise ValueError("selected snapshot must contain at least one cluster")
        selected_by_group.append((selection, clusters))
        required_keys.update(
            (context.subnarrative_id, member.candidate_nugget_id)
            for cluster in clusters
            for member in cluster.supports
        )

    canonical_by_group: dict[str, Mapping[str, object]] = {}
    for result in canonical_results:
        if result.get("topic_id") != topic.id:
            raise ValueError("canonical result topic differs from the official topic")
        group_id = result.get("subnarrative_id")
        if not isinstance(group_id, str) or group_id not in group_ids:
            raise ValueError("canonical result names an unknown selected group")
        if group_id in canonical_by_group:
            raise ValueError("duplicate canonical result for selected group")
        canonical_by_group[group_id] = result

    if original_narrative_fallback and (
        len(selections) != 1
        or selections[0].context.subnarrative_text != topic.narrative
    ):
        raise ValueError(
            "official narrative fallback requires one narrative-identical selection"
        )

    if set(snapshot.candidates) != required_keys:
        raise ValueError(
            "validated candidate projection differs from selected snapshot supports"
        )
    candidates = snapshot.candidates
    groups: list[EvidenceGroup] = []
    evidence: list[EvidencePassage] = []
    evidence_ids: set[str] = set()
    evidence_group_by_id: dict[str, str] = {}
    for selection, clusters in selected_by_group:
        context = selection.context
        projected_clusters: list[SelectedCluster] = []
        for cluster_ordinal, cluster in enumerate(clusters, start=1):
            cluster_evidence_ids: list[str] = []
            for support_ordinal, member in enumerate(cluster.supports, start=1):
                key = (context.subnarrative_id, member.candidate_nugget_id)
                candidate = candidates.get(key)
                if candidate is None or not _candidate_matches_member(
                    candidate,
                    member,
                    subnarrative_id=context.subnarrative_id,
                    subnarrative_sha256=context.subnarrative_sha256,
                ):
                    raise ValueError(
                        "selected support differs from its sealed topic-record candidate"
                    )
                evidence_id = candidate.candidate_nugget_id
                if evidence_id in evidence_ids:
                    raise ValueError("selected evidence is reused across clusters")
                rank = ranks.get(candidate.docid)
                if rank is None:
                    raise ValueError(
                        f"selected evidence document lacks retrieval rank: {candidate.docid}"
                    )
                evidence_ids.add(evidence_id)
                evidence_group_by_id[evidence_id] = context.subnarrative_id
                cluster_evidence_ids.append(evidence_id)
                evidence.append(
                    EvidencePassage(
                        evidence_id=evidence_id,
                        group_id=context.subnarrative_id,
                        cluster_id=cluster.cluster_id,
                        cluster_ordinal=cluster_ordinal,
                        support_ordinal=support_ordinal,
                        candidate_kind=candidate.candidate_kind,
                        docid=candidate.docid,
                        document_rank=rank,
                        text=candidate.text,
                        document_sha256=candidate.document_sha256,
                        source_span=_source_span(candidate),
                    )
                )
            projected_clusters.append(
                SelectedCluster(
                    cluster_id=cluster.cluster_id,
                    ordinal=cluster_ordinal,
                    representative_evidence_id=(
                        cluster.representative_candidate_nugget_id
                    ),
                    evidence_ids=tuple(cluster_evidence_ids),
                )
            )
        groups.append(
            EvidenceGroup(
                group_id=context.subnarrative_id,
                kind=(
                    "official_narrative_fallback"
                    if original_narrative_fallback
                    else "generated_subnarrative"
                ),
                text=context.subnarrative_text,
                selected_clusters=tuple(projected_clusters),
            )
        )

    claim_hints: list[ClaimHint] = []
    for selection, _clusters in selected_by_group:
        result = canonical_by_group.get(selection.context.subnarrative_id)
        if result is None:
            continue
        request = build_canonical_nugget_request(
            selection,
            selected_budget,
            max_canonical_claims=max_canonical_claims,
            max_supporting_documents_per_claim=(
                max_supporting_documents_per_claim
            ),
        )
        validate_canonical_nugget_result(result, request)
        nuggets = result["nuggets"]
        if not isinstance(nuggets, list):  # pragma: no cover - validator invariant
            raise ValueError("canonical result nuggets must be an array")
        for nugget in nuggets:
            if not isinstance(nugget, Mapping):  # pragma: no cover - validator invariant
                raise ValueError("canonical nugget must be an object")
            raw_evidence = nugget["evidence"]
            if not isinstance(raw_evidence, list):  # pragma: no cover
                raise ValueError("canonical nugget evidence must be an array")
            linked = tuple(
                item["candidate_nugget_id"]
                for item in raw_evidence
                if isinstance(item, Mapping)
            )
            if (
                len(linked) != len(raw_evidence)
                or any(
                    not isinstance(evidence_id, str)
                    or evidence_group_by_id.get(evidence_id)
                    != selection.context.subnarrative_id
                    for evidence_id in linked
                )
            ):
                raise ValueError("canonical claim links evidence outside its group")
            claim_hints.append(
                ClaimHint(
                    claim_id=nugget["canonical_nugget_id"],
                    group_id=selection.context.subnarrative_id,
                    kind=nugget["nugget_kind"],
                    text=nugget["claim_text"],
                    evidence_ids=linked,
                )
            )

    return GenerationTopic(
        topic_id=topic.id,
        narrative=topic.narrative,
        groups=tuple(groups),
        evidence=tuple(evidence),
        claim_hints=tuple(claim_hints),
        source_receipts=TopicSourceReceipts(
            official_topics_sha256=snapshot.official_topics_sha256,
            retrieval_topic_sha256=snapshot.retrieval_topic_sha256,
        ),
    )


__all__ = [
    "PreparedGenerationHandoffArtifact",
    "ValidatedGenerationSnapshot",
    "prepare_generation_handoff_artifact",
    "project_generation_topic",
]
