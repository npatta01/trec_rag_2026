"""Evidence selection for RAG generation."""

from __future__ import annotations

from trec_rag.pipeline_models import EvidenceRecord, RankedCandidate


def select_top_k_evidence(
    ranked: list[RankedCandidate],
    *,
    k: int,
    require_text: bool,
    allow_fewer: bool,
) -> list[EvidenceRecord]:
    evidence: list[EvidenceRecord] = []
    for candidate in sorted(ranked, key=lambda row: row.rank):
        if require_text and not candidate.text.strip():
            continue
        evidence.append(
            EvidenceRecord(
                topic_id=candidate.topic_id,
                docid=candidate.docid,
                citation_index=len(evidence),
                rank=candidate.rank,
                score=candidate.score,
                text=candidate.text,
                provenance=candidate.provenance,
            )
        )
        if len(evidence) == k:
            break
    if len(evidence) < k and not allow_fewer:
        raise ValueError(f"selected {len(evidence)} evidence items, fewer than required k={k}")
    return evidence
