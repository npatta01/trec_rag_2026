"""Diversity-constrained passage selection across a reranked document pool.

The agent used to choose a document and then read passages from it. Nothing
forced a second document open, and in practice none was: a measured run drew 63
grounded nuggets from 5 documents, one document per need. This module is the
half of the fix that decides *which passages the agent sees*, so breadth is a
property of the code rather than a request made of the model.

Selection is deterministic. Given the same scored chunks and the same config it
returns the same passages in the same order, so a run can be reproduced and a
regression can be attributed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from trec_rag.chunking import TextChunk

# Retrieval pool depth. Recall of answering documents (UMBRELA grade >= 3) on
# the judged dev topics climbs from 0.016 at depth 10 to 0.326 at depth 1000
# with no saturation in reach, and one request costs no extra rate-limit slot.
DEFAULT_POOL_HITS = 1000

# How many of the pooled documents get chunked and cross-encoder scored.
# Measured on this ROCm host against a real depth-1000 pool: ~12.6 chunks per
# document and ~10ms per chunk, so depth 100 costs ~14s and the full pool
# ~113s. Depth is a time budget, not a recall ceiling; recall keeps climbing.
DEFAULT_RERANK_DEPTH = 100

DEFAULT_TOP_K = 16
DEFAULT_PER_DOCUMENT_CAP = 3
DEFAULT_MIN_DISTINCT_DOCUMENTS = 6


class PassageSelectionError(ValueError):
    """Raised for an unusable selection configuration."""


@dataclass(frozen=True)
class PassageSelectionConfig:
    pool_hits: int = DEFAULT_POOL_HITS
    rerank_depth: int = DEFAULT_RERANK_DEPTH
    top_k: int = DEFAULT_TOP_K
    per_document_cap: int = DEFAULT_PER_DOCUMENT_CAP
    min_distinct_documents: int = DEFAULT_MIN_DISTINCT_DOCUMENTS

    def __post_init__(self) -> None:
        for name in (
            "pool_hits",
            "rerank_depth",
            "top_k",
            "per_document_cap",
            "min_distinct_documents",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PassageSelectionError(f"{name} must be a positive integer")
        if self.rerank_depth > self.pool_hits:
            raise PassageSelectionError("rerank_depth must not exceed pool_hits")
        if self.per_document_cap > self.top_k:
            raise PassageSelectionError("per_document_cap must not exceed top_k")
        # A breadth target the cap cannot reach would silently never bind.
        if self.min_distinct_documents > self.top_k:
            raise PassageSelectionError(
                "min_distinct_documents must not exceed top_k"
            )


@dataclass(frozen=True)
class ScoredPassage:
    """One cross-encoder scored chunk, carrying the provenance it was scored with."""

    document_id: str
    document_rank: int
    chunk: TextChunk
    relevance_score: float

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


def _ordering_key(passage: ScoredPassage) -> tuple[float, int, str]:
    """Rank by score, breaking ties by retrieval rank then chunk id.

    Ties are broken deterministically rather than by input order so selection
    does not depend on how the scoring happened to be batched.
    """
    return (-passage.relevance_score, passage.document_rank, passage.chunk_id)


def select_diverse_passages(
    passages: Iterable[ScoredPassage],
    config: PassageSelectionConfig | None = None,
) -> tuple[ScoredPassage, ...]:
    """Pick a top-K passage set that spans documents instead of collapsing onto one.

    Two phases, both deterministic:

    1. **Breadth.** Take each document's single best passage, best documents
       first, until ``min_distinct_documents`` documents are represented or the
       pool runs out of documents.
    2. **Depth.** Fill the remaining slots in global score order, admitting no
       more than ``per_document_cap`` passages from any one document.

    A plain global top-K is what collapses onto a single document, so the
    breadth phase runs first and the cap binds throughout.
    """
    resolved = config or PassageSelectionConfig()
    ordered = sorted(passages, key=_ordering_key)
    if not ordered:
        return ()

    best_by_document: dict[str, ScoredPassage] = {}
    for passage in ordered:
        best_by_document.setdefault(passage.document_id, passage)

    selected: list[ScoredPassage] = []
    per_document: dict[str, int] = {}
    taken: set[str] = set()

    breadth_target = min(
        resolved.min_distinct_documents, len(best_by_document), resolved.top_k
    )
    for passage in sorted(best_by_document.values(), key=_ordering_key):
        if len(selected) >= breadth_target:
            break
        selected.append(passage)
        per_document[passage.document_id] = 1
        taken.add(passage.chunk_id)

    for passage in ordered:
        if len(selected) >= resolved.top_k:
            break
        if passage.chunk_id in taken:
            continue
        if per_document.get(passage.document_id, 0) >= resolved.per_document_cap:
            continue
        selected.append(passage)
        per_document[passage.document_id] = per_document.get(passage.document_id, 0) + 1
        taken.add(passage.chunk_id)

    return tuple(sorted(selected, key=_ordering_key))


@dataclass(frozen=True)
class PooledDocument:
    """One retrieved document as the passage stage consumes it."""

    document_id: str
    rank: int
    text: str


@dataclass(frozen=True)
class PoolScoringResult:
    passages: tuple[ScoredPassage, ...]
    chunks_scored: int
    documents_scored: int
    documents_skipped: int


def score_document_pool(
    documents: Sequence[PooledDocument],
    focus_query: str,
    *,
    chunker,
    ranker,
    config: PassageSelectionConfig | None = None,
) -> PoolScoringResult:
    """Chunk and cross-encoder score the pool, as deep as ``rerank_depth`` allows.

    Scoring is one batched ranker call over every chunk in the scored slice, so
    passages from different documents are comparable on a single scale. Ranking
    each document separately would only produce per-document orderings that
    cannot be merged.

    Documents beyond ``rerank_depth`` are reported as skipped rather than
    dropped silently, because a bounded scan must not read as an exhaustive one.
    """
    resolved = config or PassageSelectionConfig()
    if not isinstance(focus_query, str) or not focus_query.strip():
        raise PassageSelectionError("focus_query must be non-empty text")

    ordered = sorted(documents, key=lambda item: item.rank)
    scored_slice = ordered[: resolved.rerank_depth]

    chunk_owner: dict[str, PooledDocument] = {}
    chunks: list[TextChunk] = []
    for document in scored_slice:
        if not document.text or not document.text.strip():
            continue
        for chunk in chunker.split_text(document.text, document_id=document.document_id):
            chunk_owner[chunk.chunk_id] = document
            chunks.append(chunk)

    if not chunks:
        return PoolScoringResult((), 0, len(scored_slice), len(ordered) - len(scored_slice))

    ranked = ranker.rank(focus_query, chunks)
    passages = tuple(
        ScoredPassage(
            document_id=chunk_owner[row.chunk.chunk_id].document_id,
            document_rank=chunk_owner[row.chunk.chunk_id].rank,
            chunk=row.chunk,
            relevance_score=row.relevance_score,
        )
        for row in ranked
        if row.chunk.chunk_id in chunk_owner
    )
    return PoolScoringResult(
        passages=passages,
        chunks_scored=len(chunks),
        documents_scored=len(scored_slice),
        documents_skipped=len(ordered) - len(scored_slice),
    )


def selection_summary(
    selected: Sequence[ScoredPassage],
    scored_total: int,
    documents_scored: int,
) -> dict[str, object]:
    """Describe what the selection covered, for the ledger and for traces.

    Reports what was dropped as well as what was kept, so a capped selection is
    never mistaken for an exhaustive one.
    """
    per_document: dict[str, int] = {}
    for passage in selected:
        per_document[passage.document_id] = per_document.get(passage.document_id, 0) + 1
    return {
        "returned_passages": len(selected),
        "distinct_documents": len(per_document),
        "passages_per_document": dict(sorted(per_document.items())),
        "chunks_scored": scored_total,
        "documents_scored": documents_scored,
        "chunks_not_returned": max(0, scored_total - len(selected)),
    }


def group_by_document(
    selected: Sequence[ScoredPassage],
) -> tuple[tuple[str, tuple[ScoredPassage, ...]], ...]:
    """Group selected passages by document, keeping document and passage order.

    The document ledger records one page per document, so the passage-first
    surface still produces the per-document provenance the rest of the system
    depends on.
    """
    order: list[str] = []
    grouped: dict[str, list[ScoredPassage]] = {}
    for passage in selected:
        if passage.document_id not in grouped:
            grouped[passage.document_id] = []
            order.append(passage.document_id)
        grouped[passage.document_id].append(passage)
    return tuple(
        (document_id, tuple(grouped[document_id])) for document_id in order
    )
