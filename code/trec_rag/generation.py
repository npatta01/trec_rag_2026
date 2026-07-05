"""Placeholder RAG generation for pipeline validation."""

from __future__ import annotations

from trec_rag.pipeline_models import EvidenceRecord
from trec_rag.topics import Topic


def generate_placeholder_rag(
    topic: Topic,
    evidence: list[EvidenceRecord],
    *,
    team_id: str,
    run_id: str,
) -> dict[str, object]:
    references = [item.docid for item in evidence]
    if evidence:
        answer = [
            {
                "text": (
                    "Placeholder answer: selected retrieved evidence documents are candidates for "
                    "answer generation, but no substantive answer has been generated."
                ),
                "citations": list(range(len(evidence))),
            }
        ]
    else:
        answer = [
            {
                "text": (
                    "Placeholder answer: no text-bearing retrieved evidence was selected for this topic."
                ),
                "citations": [],
            }
        ]

    return {
        "metadata": {
            "team_id": team_id,
            "run_id": run_id,
            "type": "automatic",
            "narrative_id": topic.id,
            "title": topic.title,
            "narrative": topic.narrative,
            "prompt": "placeholder generation; not an answer-quality run",
        },
        "references": references,
        "answer": answer,
    }
