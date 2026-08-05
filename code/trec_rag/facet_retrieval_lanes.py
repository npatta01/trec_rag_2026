"""Pure projection from a validated facet plan to organizer retrieval lanes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256

from trec_rag.facet_extraction import (
    GeneratedQueryPlan,
    Subnarrative,
    render_facet_queries,
)
from trec_rag.pipeline_models import QueryVariant
from trec_rag.topics import Topic


FACET_RETRIEVAL_LANE_PROJECTOR_VERSION = "facet-retrieval-lane-projector-v1"


@dataclass(frozen=True)
class FacetRetrievalLane:
    """One organizer retrieval identity paired with one scoring identity."""

    retrieval_query: QueryVariant
    scoring_query: QueryVariant
    subnarrative_id: str | None
    bm25_query_sha256: str = field(init=False)
    semantic_query_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.retrieval_query.topic_id != self.scoring_query.topic_id
            or self.retrieval_query.variant_name != self.scoring_query.variant_name
        ):
            raise ValueError("retrieval and scoring query lane identities must match")
        object.__setattr__(
            self,
            "bm25_query_sha256",
            sha256(self.retrieval_query.query_text.encode("utf-8")).hexdigest(),
        )
        object.__setattr__(
            self,
            "semantic_query_sha256",
            sha256(self.scoring_query.query_text.encode("utf-8")).hexdigest(),
        )


def build_retrieval_lanes(
    topic: Topic,
    queries: Sequence[QueryVariant],
    subnarratives: Sequence[Subnarrative],
) -> tuple[FacetRetrievalLane, ...]:
    """Bind the original and one full-text lane per generated subnarrative."""

    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    rows = tuple(queries)
    for query in rows:
        if not isinstance(query, QueryVariant) or query.topic_id != topic.id:
            raise ValueError("all retrieval queries must belong to the topic")
    originals = [
        query
        for query in rows
        if query.variant_name == "original" and query.source_type == "original_topic"
    ]
    if len(originals) != 1 or originals[0].query_text != topic.narrative:
        raise ValueError("retrieval requires one exact official-narrative original lane")
    allowed = {"original_topic", "generated_subnarrative_bm25"}
    if any(query.source_type not in allowed for query in rows):
        raise ValueError("unsupported query source type in facet retrieval plan")
    semantic_rows = tuple(subnarratives)
    if any(
        not isinstance(row, Subnarrative) or row.topic_id != topic.id
        for row in semantic_rows
    ):
        raise ValueError("all subnarratives must belong to the topic")
    if len({row.subnarrative_id for row in semantic_rows}) != len(semantic_rows):
        raise ValueError("subnarrative IDs must be unique")
    if semantic_rows:
        admitted = render_facet_queries(
            topic,
            GeneratedQueryPlan(topic.id, semantic_rows),
        )
        if (
            admitted.used_fallback
            or admitted.subnarratives != semantic_rows
            or admitted.queries != rows
        ):
            raise ValueError("retrieval requires the complete ordered canonical query plan")
    elif rows != (
        QueryVariant(topic.id, "original", topic.narrative, "original_topic"),
    ):
        raise ValueError("retrieval requires the complete ordered canonical query plan")

    bound = [
        FacetRetrievalLane(
            retrieval_query=originals[0],
            scoring_query=QueryVariant(
                topic.id,
                "original",
                topic.narrative,
                "semantic_original",
            ),
            subnarrative_id=None,
        )
    ]
    for subnarrative in semantic_rows:
        lane_name = f"facet:{subnarrative.subnarrative_id}:text"
        bound.append(
            FacetRetrievalLane(
                retrieval_query=QueryVariant(
                    topic.id,
                    lane_name,
                    subnarrative.text,
                    "generated_subnarrative",
                ),
                scoring_query=QueryVariant(
                    topic.id,
                    lane_name,
                    subnarrative.text,
                    "generated_subnarrative",
                ),
                subnarrative_id=subnarrative.subnarrative_id,
            )
        )
    return tuple(bound)
