"""Choose and rank the documents a retrieval run submits, from the ledger.

The retrieval task asks for "all and only" the documents that are relevant and
useful as evidence, ranked by usefulness, with k chosen per narrative. Two
signals were available and they are not equal:

- the reranker score says a document *looks* on-topic;
- the ledger says a document *actually supplied* evidence an agent cited, and
  in the strongest case evidence that reached a draft answer.

Measured on one judged topic, restricted to documents the qrels cover: 38% of
ledger documents were graded "answers the question" against a 13% base rate,
and none was graded irrelevant. Documents behind a vital nugget reached 43%.
The ledger is therefore the primary signal here and the score is not used.

What is deliberately not done: no cutoff is tuned against development qrels.
On the measured run 90% of the agent's documents were outside the judgement
pool, so a set-based score there reflects pooling coverage rather than
selection quality. k falls out of where ledger evidence ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from trec_rag.deepagent_evidence import EvidenceCoverageReport


# A drafted nugget reached an answer, so its documents are proven useful rather
# than merely promising; a vital one was judged essential by the researcher
# that read it. Breadth across needs is worth less than either, and raw nugget
# count least, because one document can supply many near-identical claims.
_DRAFTED_WEIGHT = 3.0
_VITAL_WEIGHT = 2.0
_NEED_WEIGHT = 1.0
_NUGGET_WEIGHT = 0.5


@dataclass(frozen=True)
class DocumentUsefulness:
    """What one document contributed, and how strongly."""

    document_id: str
    nugget_count: int
    vital_count: int
    drafted_count: int
    need_ids: tuple[str, ...]
    mean_support_ratio: float

    @property
    def score(self) -> float:
        return (
            _DRAFTED_WEIGHT * self.drafted_count
            + _VITAL_WEIGHT * self.vital_count
            + _NEED_WEIGHT * len(self.need_ids)
            + _NUGGET_WEIGHT * self.nugget_count
            + self.mean_support_ratio
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "nugget_count": self.nugget_count,
            "vital_count": self.vital_count,
            "drafted_count": self.drafted_count,
            "need_ids": list(self.need_ids),
            "mean_support_ratio": self.mean_support_ratio,
            "score": self.score,
        }


def document_usefulness(
    report: "EvidenceCoverageReport",
) -> tuple[DocumentUsefulness, ...]:
    """Summarise each document's contribution to the ledger.

    Only documents that supplied cited evidence appear. A document that was
    retrieved, reranked, read, and produced nothing is not evidence, and the
    task asks for the documents that are useful rather than the ones that were
    examined.
    """
    drafted: set[str] = set()
    for need in report.needs:
        drafted.update(need.draft_nugget_ids)

    rows: dict[str, dict[str, object]] = {}
    for nugget in report.nuggets:
        if nugget.superseded_by is not None:
            # A superseded claim was replaced; it must not still vouch for its
            # documents.
            continue
        is_vital = nugget.importance == "vital"
        is_drafted = nugget.nugget_id in drafted
        for document_id in {reference.document_id for reference in nugget.evidence}:
            row = rows.setdefault(
                document_id,
                {"nuggets": 0, "vital": 0, "drafted": 0, "needs": set(), "ratios": []},
            )
            row["nuggets"] += 1  # type: ignore[operator]
            if is_vital:
                row["vital"] += 1  # type: ignore[operator]
            if is_drafted:
                row["drafted"] += 1  # type: ignore[operator]
            row["needs"].update(nugget.need_ids)  # type: ignore[union-attr]
            row["ratios"].append(nugget.support_ratio)  # type: ignore[union-attr]

    return tuple(
        DocumentUsefulness(
            document_id=document_id,
            nugget_count=int(row["nuggets"]),  # type: ignore[arg-type]
            vital_count=int(row["vital"]),  # type: ignore[arg-type]
            drafted_count=int(row["drafted"]),  # type: ignore[arg-type]
            need_ids=tuple(sorted(row["needs"])),  # type: ignore[arg-type]
            mean_support_ratio=(
                sum(row["ratios"]) / len(row["ratios"])  # type: ignore[arg-type]
                if row["ratios"]
                else 0.0
            ),
        )
        for document_id, row in rows.items()
    )


def rank_for_submission(
    report: "EvidenceCoverageReport",
    *,
    max_documents: int | None = None,
) -> tuple[DocumentUsefulness, ...]:
    """Rank the useful documents, most useful first.

    Deterministic: ties break by document id, so the same ledger always
    produces the same run file. ``max_documents`` is a safety valve for an
    operator, not a tuned cutoff; leaving it unset submits exactly the
    documents that supplied evidence, which is what "all and only" asks for.
    """
    rows = sorted(
        document_usefulness(report),
        key=lambda row: (-row.score, row.document_id),
    )
    if max_documents is not None:
        if max_documents <= 0:
            raise ValueError("max_documents must be positive when set")
        rows = rows[:max_documents]
    return tuple(rows)


def submission_rows(
    report: "EvidenceCoverageReport",
    *,
    topic_id: str,
    run_id: str,
    max_documents: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Build TREC run rows: ``topic_id Q0 docid rank score run_id``.

    Scores are non-increasing within a narrative, which the run-file contract
    requires, and ranks start at 1.
    """
    ranked = rank_for_submission(report, max_documents=max_documents)
    return tuple(
        {
            "topic_id": topic_id,
            "q0": "Q0",
            "docid": row.document_id,
            "rank": index,
            "score": row.score,
            "run_id": run_id,
        }
        for index, row in enumerate(ranked, start=1)
    )


def selection_summary(
    ranked: Sequence[DocumentUsefulness],
) -> dict[str, object]:
    """Describe the submitted set, including what it leans on."""
    return {
        "submitted_documents": len(ranked),
        "documents_behind_a_vital_nugget": sum(1 for row in ranked if row.vital_count),
        "documents_behind_a_drafted_nugget": sum(
            1 for row in ranked if row.drafted_count
        ),
        "documents_serving_more_than_one_need": sum(
            1 for row in ranked if len(row.need_ids) > 1
        ),
        "single_nugget_documents": sum(1 for row in ranked if row.nugget_count == 1),
    }
