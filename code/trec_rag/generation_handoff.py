"""Sealed selected-evidence handoff consumed by competition RAG generation.

The records in this module are the generation seam. They contain exact selected
passages and advisory claim hints, but no full-document or retrieval-checkpoint
representation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile


HANDOFF_SCHEMA_VERSION = "generation_handoff_manifest_v1"
SOURCE_CONTRACT = "topic_records_v4"
PROMPT_CONTRACT_VERSION = "selected_evidence_one_shot_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GROUP_KINDS = frozenset({"generated_subnarrative", "official_narrative_fallback"})
_MAX_HANDOFF_BYTES = 512 * 1024 * 1024


class HandoffIntegrityError(ValueError):
    """The handoff is malformed, inconsistent, or unauthenticated."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HandoffIntegrityError(f"handoff is not canonical JSON: {exc}") from exc


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _text_digest(value: str) -> str:
    return _digest(value.encode("utf-8"))


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HandoffIntegrityError(f"{label} must be non-empty text")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HandoffIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HandoffIntegrityError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class EvidenceSourceSpan:
    start_char: int
    end_char: int
    start_byte: int
    end_byte: int

    def __post_init__(self) -> None:
        values = (self.start_char, self.end_char, self.start_byte, self.end_byte)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise HandoffIntegrityError("evidence source offsets must be integers")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise HandoffIntegrityError("evidence character span must be non-empty")
        if self.start_byte < 0 or self.end_byte <= self.start_byte:
            raise HandoffIntegrityError("evidence byte span must be non-empty")

    def to_payload(self) -> dict[str, int]:
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
        }


@dataclass(frozen=True)
class EvidencePassage:
    evidence_id: str
    group_id: str
    cluster_id: str
    cluster_ordinal: int
    support_ordinal: int
    candidate_kind: str
    docid: str
    document_rank: int
    text: str
    document_sha256: str
    source_span: EvidenceSourceSpan
    text_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "group_id",
            "cluster_id",
            "candidate_kind",
            "docid",
            "text",
        ):
            _require_text(getattr(self, name), name)
        _require_positive_int(self.cluster_ordinal, "cluster_ordinal")
        _require_positive_int(self.support_ordinal, "support_ordinal")
        _require_positive_int(self.document_rank, "document_rank")
        _require_sha256(self.document_sha256, "document_sha256")
        if not isinstance(self.source_span, EvidenceSourceSpan):
            raise HandoffIntegrityError("source_span must be EvidenceSourceSpan")
        if self.source_span.end_char - self.source_span.start_char != len(self.text):
            raise HandoffIntegrityError("evidence character span length differs from text")
        if self.source_span.end_byte - self.source_span.start_byte != len(
            self.text.encode("utf-8")
        ):
            raise HandoffIntegrityError("evidence byte span length differs from UTF-8 text")
        object.__setattr__(self, "text_sha256", _text_digest(self.text))

    def to_payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "group_id": self.group_id,
            "cluster_id": self.cluster_id,
            "cluster_ordinal": self.cluster_ordinal,
            "support_ordinal": self.support_ordinal,
            "candidate_kind": self.candidate_kind,
            "docid": self.docid,
            "document_rank": self.document_rank,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "document_sha256": self.document_sha256,
            "source_span": self.source_span.to_payload(),
        }


@dataclass(frozen=True)
class SelectedCluster:
    cluster_id: str
    ordinal: int
    representative_evidence_id: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.cluster_id, "cluster_id")
        _require_positive_int(self.ordinal, "cluster ordinal")
        _require_text(self.representative_evidence_id, "representative_evidence_id")
        if (
            not isinstance(self.evidence_ids, tuple)
            or not self.evidence_ids
            or any(not isinstance(value, str) or not value for value in self.evidence_ids)
            or len(set(self.evidence_ids)) != len(self.evidence_ids)
        ):
            raise HandoffIntegrityError(
                "cluster evidence_ids must be unique non-empty text"
            )
        if self.representative_evidence_id not in self.evidence_ids:
            raise HandoffIntegrityError(
                "cluster representative must identify selected evidence"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "ordinal": self.ordinal,
            "representative_evidence_id": self.representative_evidence_id,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class EvidenceGroup:
    group_id: str
    kind: str
    text: str
    selected_clusters: tuple[SelectedCluster, ...]
    text_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_text(self.group_id, "group_id")
        if self.kind not in _GROUP_KINDS:
            raise HandoffIntegrityError(f"unsupported evidence group kind: {self.kind!r}")
        _require_text(self.text, "group text")
        if (
            not isinstance(self.selected_clusters, tuple)
            or not self.selected_clusters
            or any(
                not isinstance(cluster, SelectedCluster)
                for cluster in self.selected_clusters
            )
        ):
            raise HandoffIntegrityError(
                "selected_clusters must be a non-empty tuple of SelectedCluster"
            )
        expected_ordinals = tuple(range(1, len(self.selected_clusters) + 1))
        if tuple(cluster.ordinal for cluster in self.selected_clusters) != expected_ordinals:
            raise HandoffIntegrityError("cluster ordinals must be contiguous from one")
        cluster_ids = tuple(cluster.cluster_id for cluster in self.selected_clusters)
        if len(set(cluster_ids)) != len(cluster_ids):
            raise HandoffIntegrityError("cluster IDs must be unique within a group")
        object.__setattr__(self, "text_sha256", _text_digest(self.text))

    def to_payload(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "kind": self.kind,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "selected_clusters": [
                cluster.to_payload() for cluster in self.selected_clusters
            ],
        }


@dataclass(frozen=True)
class ClaimHint:
    claim_id: str
    group_id: str
    kind: str
    text: str
    evidence_ids: tuple[str, ...]
    text_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("claim_id", "group_id", "kind", "text"):
            _require_text(getattr(self, name), name)
        if (
            not isinstance(self.evidence_ids, tuple)
            or not self.evidence_ids
            or any(not isinstance(value, str) or not value for value in self.evidence_ids)
            or len(set(self.evidence_ids)) != len(self.evidence_ids)
        ):
            raise HandoffIntegrityError(
                "claim evidence_ids must be unique non-empty text"
            )
        object.__setattr__(self, "text_sha256", _text_digest(self.text))

    def to_payload(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "group_id": self.group_id,
            "kind": self.kind,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class TopicSourceReceipts:
    official_topics_sha256: str
    retrieval_topic_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.official_topics_sha256, "official_topics receipt")
        _require_sha256(self.retrieval_topic_sha256, "retrieval_topic receipt")

    def to_payload(
        self,
        *,
        evidence_payload: list[dict[str, object]],
        claim_payload: list[dict[str, object]],
    ) -> dict[str, str]:
        return {
            "official_topics": self.official_topics_sha256,
            "retrieval_topic": self.retrieval_topic_sha256,
            "selected_evidence": _digest(_canonical_json(evidence_payload)),
            "canonical_claim_hints": _digest(_canonical_json(claim_payload)),
        }


@dataclass(frozen=True)
class GenerationTopic:
    topic_id: str
    narrative: str
    groups: tuple[EvidenceGroup, ...]
    evidence: tuple[EvidencePassage, ...]
    claim_hints: tuple[ClaimHint, ...]
    source_receipts: TopicSourceReceipts
    narrative_sha256: str = field(init=False)
    citation_docids: tuple[str, ...] = field(init=False)
    context_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_text(self.topic_id, "topic_id")
        _require_text(self.narrative, "narrative")
        if (
            not isinstance(self.groups, tuple)
            or not self.groups
            or any(not isinstance(group, EvidenceGroup) for group in self.groups)
        ):
            raise HandoffIntegrityError("groups must be a non-empty tuple of EvidenceGroup")
        if (
            not isinstance(self.evidence, tuple)
            or not self.evidence
            or any(not isinstance(row, EvidencePassage) for row in self.evidence)
        ):
            raise HandoffIntegrityError(
                "evidence must be a non-empty tuple of EvidencePassage"
            )
        if not isinstance(self.claim_hints, tuple) or any(
            not isinstance(row, ClaimHint) for row in self.claim_hints
        ):
            raise HandoffIntegrityError("claim_hints must be a tuple of ClaimHint")
        if not isinstance(self.source_receipts, TopicSourceReceipts):
            raise HandoffIntegrityError("source_receipts must be TopicSourceReceipts")

        groups_by_id = {group.group_id: group for group in self.groups}
        if len(groups_by_id) != len(self.groups):
            raise HandoffIntegrityError("group IDs must be unique")
        fallback_groups = [
            group for group in self.groups if group.kind == "official_narrative_fallback"
        ]
        if fallback_groups and (
            len(self.groups) != 1 or fallback_groups[0].text != self.narrative
        ):
            raise HandoffIntegrityError(
                "official narrative fallback must be the only group and copy the narrative"
            )

        evidence_by_id = {row.evidence_id: row for row in self.evidence}
        if len(evidence_by_id) != len(self.evidence):
            raise HandoffIntegrityError("evidence IDs must be unique")
        flattened_ids: list[str] = []
        for group in self.groups:
            for cluster in group.selected_clusters:
                for support_ordinal, evidence_id in enumerate(
                    cluster.evidence_ids, start=1
                ):
                    row = evidence_by_id.get(evidence_id)
                    if row is None:
                        raise HandoffIntegrityError(
                            f"cluster references unknown evidence: {evidence_id}"
                        )
                    if (
                        row.group_id != group.group_id
                        or row.cluster_id != cluster.cluster_id
                        or row.cluster_ordinal != cluster.ordinal
                        or row.support_ordinal != support_ordinal
                    ):
                        raise HandoffIntegrityError(
                            f"evidence provenance differs from selected cluster: {evidence_id}"
                        )
                    flattened_ids.append(evidence_id)
        if tuple(flattened_ids) != tuple(row.evidence_id for row in self.evidence):
            raise HandoffIntegrityError(
                "evidence order must exactly flatten selected cluster supports"
            )

        claim_ids: set[str] = set()
        for claim in self.claim_hints:
            if claim.claim_id in claim_ids:
                raise HandoffIntegrityError("claim IDs must be unique")
            claim_ids.add(claim.claim_id)
            if claim.group_id not in groups_by_id:
                raise HandoffIntegrityError(
                    f"claim references unknown group: {claim.group_id}"
                )
            for evidence_id in claim.evidence_ids:
                row = evidence_by_id.get(evidence_id)
                if row is None or row.group_id != claim.group_id:
                    raise HandoffIntegrityError(
                        f"claim evidence is unknown or outside its group: {evidence_id}"
                    )

        document_identity: dict[str, tuple[int, str]] = {}
        rank_to_docid: dict[int, str] = {}
        for row in self.evidence:
            identity = (row.document_rank, row.document_sha256)
            existing = document_identity.setdefault(row.docid, identity)
            if existing != identity:
                raise HandoffIntegrityError(
                    f"one docid has conflicting rank or hash: {row.docid}"
                )
            ranked_docid = rank_to_docid.setdefault(row.document_rank, row.docid)
            if ranked_docid != row.docid:
                raise HandoffIntegrityError(
                    f"document rank {row.document_rank} identifies multiple docids"
                )

        object.__setattr__(self, "narrative_sha256", _text_digest(self.narrative))
        citation_docids = tuple(
            docid
            for docid, _identity in sorted(
                document_identity.items(), key=lambda item: (item[1][0], item[0])
            )
        )
        object.__setattr__(self, "citation_docids", citation_docids)
        object.__setattr__(
            self,
            "context_sha256",
            _digest(_canonical_json(self._payload_without_context_hash())),
        )

    def _payload_without_context_hash(self) -> dict[str, object]:
        evidence_payload = [row.to_payload() for row in self.evidence]
        claim_payload = [row.to_payload() for row in self.claim_hints]
        return {
            "topic_id": self.topic_id,
            "narrative": self.narrative,
            "narrative_sha256": self.narrative_sha256,
            "groups": [group.to_payload() for group in self.groups],
            "evidence": evidence_payload,
            "claim_hints": claim_payload,
            "citation_docids": list(self.citation_docids),
            "source_receipts": self.source_receipts.to_payload(
                evidence_payload=evidence_payload,
                claim_payload=claim_payload,
            ),
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_context_hash(),
            "context_sha256": self.context_sha256,
        }


@dataclass(frozen=True)
class HandoffProducer:
    source_contract: str
    retrieval_run_id: str
    producer_revision: str

    def __post_init__(self) -> None:
        if self.source_contract != SOURCE_CONTRACT:
            raise HandoffIntegrityError(
                f"source_contract must be {SOURCE_CONTRACT}"
            )
        _require_text(self.retrieval_run_id, "retrieval_run_id")
        _require_text(self.producer_revision, "producer_revision")

    def to_payload(self) -> dict[str, str]:
        return {
            "source_contract": self.source_contract,
            "retrieval_run_id": self.retrieval_run_id,
            "producer_revision": self.producer_revision,
        }


@dataclass(frozen=True)
class GenerationHandoff:
    producer: HandoffProducer
    topics: tuple[GenerationTopic, ...]
    schema_version: str = field(init=False, default=HANDOFF_SCHEMA_VERSION)
    topic_count: int = field(init=False)
    manifest_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.producer, HandoffProducer):
            raise HandoffIntegrityError("producer must be HandoffProducer")
        if (
            not isinstance(self.topics, tuple)
            or not self.topics
            or any(not isinstance(topic, GenerationTopic) for topic in self.topics)
        ):
            raise HandoffIntegrityError(
                "topics must be a non-empty tuple of GenerationTopic"
            )
        topic_ids = tuple(topic.topic_id for topic in self.topics)
        if len(set(topic_ids)) != len(topic_ids):
            raise HandoffIntegrityError("topic IDs must be unique")
        object.__setattr__(self, "topic_count", len(self.topics))
        object.__setattr__(
            self,
            "manifest_sha256",
            _digest(_canonical_json(self._payload_without_manifest_hash())),
        )

    def _payload_without_manifest_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "producer": self.producer.to_payload(),
            "topic_count": self.topic_count,
            "topics": [topic.to_payload() for topic in self.topics],
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_manifest_hash(),
            "manifest_sha256": self.manifest_sha256,
        }


def serialize_generation_handoff(handoff: GenerationHandoff) -> bytes:
    if not isinstance(handoff, GenerationHandoff):
        raise TypeError("handoff must be GenerationHandoff")
    return _canonical_json(handoff.to_payload()) + b"\n"


def serialize_generation_topic(topic: GenerationTopic) -> bytes:
    """Serialize one authenticated topic for a sealed per-topic projection."""
    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    return _canonical_json(topic.to_payload()) + b"\n"


def _exact_mapping(
    value: object,
    expected: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise HandoffIntegrityError(f"{label} must be an object")
    fields = set(value)
    if fields != expected:
        missing = sorted(expected - fields)
        unknown = sorted(fields - expected)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise HandoffIntegrityError(f"{label} fields are invalid ({'; '.join(detail)})")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise HandoffIntegrityError(f"{label} must be an array")
    return value


def _parse_span(value: object) -> EvidenceSourceSpan:
    row = _exact_mapping(
        value,
        {"start_char", "end_char", "start_byte", "end_byte"},
        "source_span",
    )
    return EvidenceSourceSpan(
        start_char=row["start_char"],  # type: ignore[arg-type]
        end_char=row["end_char"],  # type: ignore[arg-type]
        start_byte=row["start_byte"],  # type: ignore[arg-type]
        end_byte=row["end_byte"],  # type: ignore[arg-type]
    )


def _parse_evidence(value: object) -> EvidencePassage:
    row = _exact_mapping(
        value,
        {
            "evidence_id",
            "group_id",
            "cluster_id",
            "cluster_ordinal",
            "support_ordinal",
            "candidate_kind",
            "docid",
            "document_rank",
            "text",
            "text_sha256",
            "document_sha256",
            "source_span",
        },
        "evidence payload",
    )
    evidence = EvidencePassage(
        evidence_id=row["evidence_id"],  # type: ignore[arg-type]
        group_id=row["group_id"],  # type: ignore[arg-type]
        cluster_id=row["cluster_id"],  # type: ignore[arg-type]
        cluster_ordinal=row["cluster_ordinal"],  # type: ignore[arg-type]
        support_ordinal=row["support_ordinal"],  # type: ignore[arg-type]
        candidate_kind=row["candidate_kind"],  # type: ignore[arg-type]
        docid=row["docid"],  # type: ignore[arg-type]
        document_rank=row["document_rank"],  # type: ignore[arg-type]
        text=row["text"],  # type: ignore[arg-type]
        document_sha256=row["document_sha256"],  # type: ignore[arg-type]
        source_span=_parse_span(row["source_span"]),
    )
    if evidence.to_payload() != dict(row):
        raise HandoffIntegrityError(
            f"evidence payload hash or derived fields are invalid: {evidence.evidence_id}"
        )
    return evidence


def _parse_cluster(value: object) -> SelectedCluster:
    row = _exact_mapping(
        value,
        {"cluster_id", "ordinal", "representative_evidence_id", "evidence_ids"},
        "selected cluster",
    )
    evidence_ids = _list(row["evidence_ids"], "selected cluster evidence_ids")
    cluster = SelectedCluster(
        cluster_id=row["cluster_id"],  # type: ignore[arg-type]
        ordinal=row["ordinal"],  # type: ignore[arg-type]
        representative_evidence_id=row["representative_evidence_id"],  # type: ignore[arg-type]
        evidence_ids=tuple(evidence_ids),  # type: ignore[arg-type]
    )
    if cluster.to_payload() != dict(row):
        raise HandoffIntegrityError("selected cluster payload is invalid")
    return cluster


def _parse_group(value: object) -> EvidenceGroup:
    row = _exact_mapping(
        value,
        {"group_id", "kind", "text", "text_sha256", "selected_clusters"},
        "evidence group",
    )
    clusters = tuple(
        _parse_cluster(item)
        for item in _list(row["selected_clusters"], "selected_clusters")
    )
    group = EvidenceGroup(
        group_id=row["group_id"],  # type: ignore[arg-type]
        kind=row["kind"],  # type: ignore[arg-type]
        text=row["text"],  # type: ignore[arg-type]
        selected_clusters=clusters,
    )
    if group.to_payload() != dict(row):
        raise HandoffIntegrityError(
            f"evidence group payload is invalid: {group.group_id}"
        )
    return group


def _parse_claim(value: object) -> ClaimHint:
    row = _exact_mapping(
        value,
        {"claim_id", "group_id", "kind", "text", "text_sha256", "evidence_ids"},
        "claim hint",
    )
    claim = ClaimHint(
        claim_id=row["claim_id"],  # type: ignore[arg-type]
        group_id=row["group_id"],  # type: ignore[arg-type]
        kind=row["kind"],  # type: ignore[arg-type]
        text=row["text"],  # type: ignore[arg-type]
        evidence_ids=tuple(_list(row["evidence_ids"], "claim evidence_ids")),  # type: ignore[arg-type]
    )
    if claim.to_payload() != dict(row):
        raise HandoffIntegrityError(f"claim hint payload is invalid: {claim.claim_id}")
    return claim


def _parse_topic(value: object) -> GenerationTopic:
    row = _exact_mapping(
        value,
        {
            "topic_id",
            "narrative",
            "narrative_sha256",
            "groups",
            "evidence",
            "claim_hints",
            "citation_docids",
            "source_receipts",
            "context_sha256",
        },
        "topic",
    )
    receipts = _exact_mapping(
        row["source_receipts"],
        {
            "official_topics",
            "retrieval_topic",
            "selected_evidence",
            "canonical_claim_hints",
        },
        "source_receipts",
    )
    topic = GenerationTopic(
        topic_id=row["topic_id"],  # type: ignore[arg-type]
        narrative=row["narrative"],  # type: ignore[arg-type]
        groups=tuple(
            _parse_group(item) for item in _list(row["groups"], "groups")
        ),
        evidence=tuple(
            _parse_evidence(item) for item in _list(row["evidence"], "evidence")
        ),
        claim_hints=tuple(
            _parse_claim(item)
            for item in _list(row["claim_hints"], "claim_hints")
        ),
        source_receipts=TopicSourceReceipts(
            official_topics_sha256=receipts["official_topics"],  # type: ignore[arg-type]
            retrieval_topic_sha256=receipts["retrieval_topic"],  # type: ignore[arg-type]
        ),
    )
    if topic.to_payload() != dict(row):
        raise HandoffIntegrityError(
            f"topic payload hashes or derived fields are invalid: {topic.topic_id}"
        )
    return topic


def _parse_producer(value: object) -> HandoffProducer:
    row = _exact_mapping(
        value,
        {"source_contract", "retrieval_run_id", "producer_revision"},
        "producer",
    )
    producer = HandoffProducer(
        source_contract=row["source_contract"],  # type: ignore[arg-type]
        retrieval_run_id=row["retrieval_run_id"],  # type: ignore[arg-type]
        producer_revision=row["producer_revision"],  # type: ignore[arg-type]
    )
    if producer.to_payload() != dict(row):
        raise HandoffIntegrityError("producer payload is invalid")
    return producer


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HandoffIntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise HandoffIntegrityError(f"invalid non-finite JSON value: {value}")


def deserialize_generation_topic(source: bytes) -> GenerationTopic:
    """Strictly reconstruct one canonical per-topic generation projection."""
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    if len(source) > _MAX_HANDOFF_BYTES:
        raise HandoffIntegrityError("generation topic projection is too large")
    try:
        payload = json.loads(
            source.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except HandoffIntegrityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandoffIntegrityError(
            f"invalid generation topic projection: {exc}"
        ) from exc
    topic = _parse_topic(payload)
    if source != serialize_generation_topic(topic):
        raise HandoffIntegrityError(
            "generation topic projection bytes are not canonical"
        )
    return topic


def load_generation_handoff(path: Path) -> GenerationHandoff:
    """Load and fully authenticate one self-contained generation handoff."""
    path = Path(path)
    try:
        if path.stat().st_size > _MAX_HANDOFF_BYTES:
            raise HandoffIntegrityError(
                f"handoff exceeds {_MAX_HANDOFF_BYTES} bytes: {path}"
            )
        source = path.read_bytes()
    except HandoffIntegrityError:
        raise
    except OSError as exc:
        raise HandoffIntegrityError(f"cannot read handoff {path}: {exc}") from exc
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HandoffIntegrityError(f"handoff is not valid UTF-8: {path}") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except HandoffIntegrityError:
        raise
    except json.JSONDecodeError as exc:
        raise HandoffIntegrityError(f"invalid handoff JSON: {exc}") from exc

    root = _exact_mapping(
        payload,
        {"schema_version", "producer", "topic_count", "topics", "manifest_sha256"},
        "root",
    )
    if root["schema_version"] != HANDOFF_SCHEMA_VERSION:
        raise HandoffIntegrityError(
            f"schema_version must be {HANDOFF_SCHEMA_VERSION}"
        )
    _require_sha256(root["manifest_sha256"], "manifest_sha256")
    topics = tuple(
        _parse_topic(item) for item in _list(root["topics"], "topics")
    )
    if (
        isinstance(root["topic_count"], bool)
        or not isinstance(root["topic_count"], int)
        or root["topic_count"] != len(topics)
    ):
        raise HandoffIntegrityError("topic_count does not match topics")
    handoff = GenerationHandoff(
        producer=_parse_producer(root["producer"]),
        topics=topics,
    )
    if handoff.to_payload() != dict(root):
        raise HandoffIntegrityError("root manifest hash or derived fields are invalid")
    if source != serialize_generation_handoff(handoff):
        raise HandoffIntegrityError(
            "handoff bytes are not the canonical manifest serialization"
        )
    return handoff


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_generation_handoff(path: Path, handoff: GenerationHandoff) -> None:
    """Publish canonical handoff bytes without replacing contradictory state."""
    path = Path(path)
    contents = serialize_generation_handoff(handoff)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise HandoffIntegrityError(f"handoff destination is a symbolic link: {path}")
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise HandoffIntegrityError(f"cannot read existing handoff {path}: {exc}") from exc
        if existing == contents:
            return
        raise HandoffIntegrityError(f"handoff destination contains different bytes: {path}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != contents:
                raise HandoffIntegrityError(
                    f"handoff destination contains different bytes: {path}"
                ) from None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def select_generation_topics(
    handoff: GenerationHandoff,
    topic_ids: Sequence[str] | None,
) -> tuple[GenerationTopic, ...]:
    """Select unique known topic IDs while retaining manifest order."""
    if not isinstance(handoff, GenerationHandoff):
        raise TypeError("handoff must be GenerationHandoff")
    if topic_ids is None:
        return handoff.topics
    if isinstance(topic_ids, (str, bytes)) or not isinstance(topic_ids, Sequence):
        raise HandoffIntegrityError("topic_ids must be a sequence of topic IDs")
    if not topic_ids:
        raise HandoffIntegrityError("topic_ids must contain at least one topic ID")
    requested: set[str] = set()
    available = {topic.topic_id for topic in handoff.topics}
    for topic_id in topic_ids:
        if not isinstance(topic_id, str) or not topic_id:
            raise HandoffIntegrityError("topic_ids must contain non-empty text")
        if topic_id in requested:
            raise HandoffIntegrityError(f"duplicate topic ID: {topic_id}")
        if topic_id not in available:
            raise HandoffIntegrityError(f"unknown topic ID: {topic_id}")
        requested.add(topic_id)
    return tuple(topic for topic in handoff.topics if topic.topic_id in requested)


def render_generation_evidence(topic: GenerationTopic) -> str:
    """Render only the selected factual passages and their advisory routing hints."""
    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    evidence_by_id = {row.evidence_id: row for row in topic.evidence}
    claims_by_group: dict[str, list[ClaimHint]] = {
        group.group_id: [] for group in topic.groups
    }
    for claim in topic.claim_hints:
        claims_by_group[claim.group_id].append(claim)

    lines = [
        "FROZEN SELECTED RETRIEVAL EVIDENCE",
        f"topic_id: {topic.topic_id}",
        "OFFICIAL NARRATIVE:",
        topic.narrative,
        "",
        (
            "Selected evidence passages are factual authority. Claim hints are "
            "advisory routing aids; include a claim only when the linked passage "
            "text supports it."
        ),
    ]
    for group in topic.groups:
        lines.extend(
            [
                "",
                f"GROUP group_id={group.group_id} kind={group.kind}",
                f"group_text: {group.text}",
            ]
        )
        for cluster in group.selected_clusters:
            lines.append(
                f"CLUSTER ordinal={cluster.ordinal} cluster_id={cluster.cluster_id}"
            )
            for evidence_id in cluster.evidence_ids:
                evidence = evidence_by_id[evidence_id]
                lines.extend(
                    [
                        (
                            "[EVIDENCE "
                            f"evidence_id={evidence.evidence_id} "
                            f"docid={evidence.docid} "
                            f"document_rank={evidence.document_rank} "
                            f"group_id={evidence.group_id} "
                            f"cluster_id={evidence.cluster_id}]"
                        ),
                        evidence.text,
                    ]
                )
        lines.append("ADVISORY CLAIM HINTS:")
        claims = claims_by_group[group.group_id]
        if not claims:
            lines.append("(none)")
        for claim in claims:
            lines.append(
                f"- claim_id={claim.claim_id} kind={claim.kind} "
                f"evidence_ids={','.join(claim.evidence_ids)}: {claim.text}"
            )
    return "\n".join(lines)


def prompt_sha256(topic: GenerationTopic) -> str:
    """Authenticate the exact UTF-8 evidence rendering used in a writer prompt."""
    return _text_digest(render_generation_evidence(topic))


__all__ = [
    "HANDOFF_SCHEMA_VERSION",
    "PROMPT_CONTRACT_VERSION",
    "SOURCE_CONTRACT",
    "ClaimHint",
    "EvidenceGroup",
    "EvidencePassage",
    "EvidenceSourceSpan",
    "GenerationHandoff",
    "GenerationTopic",
    "HandoffIntegrityError",
    "HandoffProducer",
    "SelectedCluster",
    "TopicSourceReceipts",
    "deserialize_generation_topic",
    "load_generation_handoff",
    "prompt_sha256",
    "render_generation_evidence",
    "select_generation_topics",
    "serialize_generation_handoff",
    "serialize_generation_topic",
    "write_generation_handoff",
]
