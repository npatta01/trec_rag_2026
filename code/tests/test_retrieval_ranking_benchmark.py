from __future__ import annotations

from pathlib import Path

import pytest

from trec_rag.retrieval_ranking_benchmark import (
    load_benchmark_config,
    paired_statistics,
    validate_modal_budget,
)


ROOT = Path(__file__).resolve().parents[2]


def test_checked_in_config_is_valid_and_cost_is_below_ten_dollars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TREC_RAG_SHARED_ROOT", str(ROOT))
    config = load_benchmark_config(ROOT / "configs" / "rag25_pointwise_listwise_ndcg_v1.yaml")
    budget = validate_modal_budget(config.raw["modal_budget"])

    assert config.raw["organizer_pointwise"]["dtype"] == "bfloat16"
    assert config.raw["organizer_listwise"]["dtype"] == "bfloat16"
    assert budget["total_maximum_usd"] < 10.0


def test_budget_validation_fails_closed_at_cap() -> None:
    settings = {
        "maximum_usd": 1,
        "prior_spend_usd": 0,
        "a100_80gb_usd_per_second": 1,
        "cpu_usd_per_core_second": 0,
        "memory_usd_per_gib_second": 0,
        "gpu_cpu_cores": 0,
        "gpu_memory_gib": 0,
        "smoke_timeout_seconds": 1,
        "full_timeout_seconds": 1,
        "staging_cpu_cores": 0,
        "staging_memory_gib": 0,
        "staging_timeout_seconds": 0,
    }
    with pytest.raises(ValueError, match="not below"):
        validate_modal_budget(settings)


def test_paired_statistics_are_deterministic_and_paired() -> None:
    first = paired_statistics(
        [0.4, 0.7, 0.6],
        [0.2, 0.5, 0.6],
        bootstrap_samples=1000,
        randomization_samples=1000,
        seed=17,
    )
    second = paired_statistics(
        [0.4, 0.7, 0.6],
        [0.2, 0.5, 0.6],
        bootstrap_samples=1000,
        randomization_samples=1000,
        seed=17,
    )

    assert first == second
    assert first["mean_delta"] == pytest.approx(0.4 / 3)
    assert (first["wins"], first["ties"], first["losses"]) == (2, 1, 0)
