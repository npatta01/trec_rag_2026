from __future__ import annotations

import pytest

from trec_rag.topic213_2026_submission import validate_submission_entry


def _entry():
    return {
        "metadata": {
            "team_id": "team",
            "narrative_id": "213",
            "narrative": "A narrative",
            "run_id": "run",
            "run_desc": "description",
        },
        "references": ["shard_00001_1", "shard_00002_2"],
        "answer": [
            {
                "text": "A synthesized sentence supported by two documents.",
                "citations": ["shard_00001_1", "shard_00002_2"],
            }
        ],
    }


def test_validate_submission_accepts_direct_climbmix_citations():
    assert validate_submission_entry(_entry()) == 7


def test_validate_submission_rejects_more_than_three_citations():
    entry = _entry()
    entry["references"] += ["shard_00003_3", "shard_00004_4"]
    entry["answer"][0]["citations"] = entry["references"]
    with pytest.raises(ValueError, match="one to three"):
        validate_submission_entry(entry)


def test_validate_submission_rejects_non_climbmix_reference():
    entry = _entry()
    entry["references"][0] = "P000001"
    with pytest.raises(ValueError, match="ClimbMix"):
        validate_submission_entry(entry)
