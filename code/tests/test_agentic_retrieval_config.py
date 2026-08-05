from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

from trec_rag.facet_pilot_config import load_facet_pilot_config


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONFIG = ROOT / "configs" / "rag26_competition_agentic_retrieval_v1.yaml"
FIXED_CONFIG = ROOT / "configs" / "rag26_competition_retrieval_v2.yaml"
TOPICS = ROOT / "trec-rag-data" / "trec-rag-2026" / "test-data" / "trec_rag_2026_queries.tsv"
CORPUS_EPOCH = (
    "climbmix-400b-operator-archive-facet-2025-v2-sha256-"
    "2ccad901eb908ac14747bafd2069f7c8aa96ecaaa8cc30457ff47c73ce3bf81f"
)
MODEL_REVISION = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"


def _agentic_module():
    try:
        return import_module("trec_rag.agentic_retrieval_config")
    except ModuleNotFoundError:
        pytest.fail("agentic retrieval config module is missing")


def _config_text(*, experiment_id: str = "agentic-test") -> str:
    return f"""\
schema_version: agentic_retrieval_config_v1
retrieval_mode: agentic

experiment:
  id: {experiment_id}

topics:
  path: {TOPICS}

execution:
  topic_workers: 1

caches:
  document_store_dir: cache/documents/v1
  model_cache_dir: cache/models/huggingface

retrieval:
  index: climbmix-400b
  cache_dir: cache/retrieval/pyserini_remote
  documents_per_query: 1000
  hits_per_search: 10
  corpus_epoch: {CORPUS_EPOCH}

passage:
  model: mixedbread-ai/mxbai-rerank-base-v2
  revision: {MODEL_REVISION}
  score_cache_dir: cache/reranker
  device: auto
  passages_per_query: 100
  chunk_max_characters: 3500
  chunk_overlap_characters: 350

snippets:
  result_cache_dir: cache/reranker/deepagent_snippets
  snippets_per_page: 10

models:
  coordinator_and_researcher: openrouter:deepseek/deepseek-v4-flash

agent:
  fused_result_limit: 20

budget:
  max_researcher_invocations: 20
  max_concurrent: 3
  max_retrieval_calls: 100
  max_tools_per_researcher: 20
  max_searches_per_researcher: 8
  max_snippets_per_researcher: 16
  max_passage_searches_per_researcher: 8
  max_models_per_researcher: 30
  max_main_models: 80
  synthesis_reserve_turns: 2
  no_yield_calls: 3
  no_progress_rounds: 2
"""


def _write_config(tmp_path: Path, text: str | None = None) -> Path:
    path = tmp_path / "agentic.yaml"
    path.write_text(text or _config_text(), encoding="utf-8")
    return path


def test_canonical_agentic_config_loads_the_full_official_cohort() -> None:
    module = _agentic_module()

    config = module.load_agentic_retrieval_config(CANONICAL_CONFIG)
    topics = module.select_agentic_topics(config)

    assert config.retrieval_mode == "agentic"
    assert len(topics) == 119
    assert topics[0].id == "rag2026-0"
    assert topics[-1].id == "rag2026-118"
    assert config.output_dir == config.root_dir / "outputs" / config.run_id


def test_canonical_config_pins_the_approved_production_limits() -> None:
    module = _agentic_module()

    config = module.load_agentic_retrieval_config(CANONICAL_CONFIG)

    assert config.execution.topic_workers == 1
    assert config.retrieval.documents_per_query == 1000
    assert config.retrieval.hits_per_search == 10
    assert config.agent.fused_result_limit == 20
    assert config.budget.max_researcher_invocations == 20
    assert config.budget.max_concurrent == 3
    assert config.budget.max_retrieval_calls == 100
    assert config.budget.max_tools_per_researcher == 20
    assert config.budget.max_searches_per_researcher == 8
    assert config.budget.max_snippets_per_researcher == 16
    assert config.budget.max_passage_searches_per_researcher == 8
    assert config.budget.max_models_per_researcher == 30
    assert config.budget.max_main_models == 80
    assert config.budget.synthesis_reserve_turns == 2
    assert config.budget.no_yield_calls == 3
    assert config.budget.no_progress_rounds == 2
    assert config.budget.soft_seconds is None
    assert config.budget.hard_seconds is None


def test_agentic_and_fixed_loaders_reject_the_other_mode() -> None:
    module = _agentic_module()

    with pytest.raises(ValueError, match="schema_version"):
        module.load_agentic_retrieval_config(FIXED_CONFIG)
    with pytest.raises(ValueError, match="schema_version"):
        load_facet_pilot_config(CANONICAL_CONFIG)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_config_text() + "unknown: true\n", "unknown field"),
        (
            _config_text().replace(
                "  topic_workers: 1\n",
                "  topic_workers: 1\n  topic_workers: 1\n",
            ),
            "duplicate YAML key",
        ),
        (_config_text(experiment_id="../escape"), "safe identifier"),
    ],
)
def test_config_rejects_unknown_duplicate_and_unsafe_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    message: str,
) -> None:
    module = _agentic_module()
    monkeypatch.setattr(module, "find_repo_root", lambda _start: ROOT)

    with pytest.raises(ValueError, match=message):
        module.load_agentic_retrieval_config(_write_config(tmp_path, text))


def test_cache_paths_resolve_through_the_shared_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _agentic_module()
    shared_cache = tmp_path / "shared" / "cache"
    monkeypatch.setattr(module, "find_repo_root", lambda _start: ROOT)
    monkeypatch.setattr(module, "repo_cache_root", lambda _root: shared_cache)

    config = module.load_agentic_retrieval_config(_write_config(tmp_path))

    assert config.retrieval.cache_dir == shared_cache / "retrieval" / "pyserini_remote"
    assert config.passage.score_cache_dir == shared_cache / "reranker"
    assert config.snippets.result_cache_dir == shared_cache / "reranker" / "deepagent_snippets"
    assert config.caches.document_store_dir == shared_cache / "documents" / "v1"
    assert config.caches.model_cache_dir == shared_cache / "models" / "huggingface"


def test_topic_selectors_preserve_official_order_and_reject_duplicates() -> None:
    module = _agentic_module()
    config = module.load_agentic_retrieval_config(CANONICAL_CONFIG)

    selected = module.select_agentic_topics(
        config,
        topic_ids=("rag2026-2", "rag2026-0", "rag2026-1"),
    )

    assert [topic.id for topic in selected] == [
        "rag2026-0",
        "rag2026-1",
        "rag2026-2",
    ]
    with pytest.raises(ValueError, match="duplicate topic ID"):
        module.select_agentic_topics(
            config,
            topic_ids=("rag2026-0", "rag2026-0"),
        )


def test_resolved_payload_is_non_secret_and_binds_selected_topics() -> None:
    module = _agentic_module()
    config = module.load_agentic_retrieval_config(CANONICAL_CONFIG)
    selected = module.select_agentic_topics(config, topic_ids=("rag2026-0",))

    payload = config.resolved_payload(selected)

    assert payload["schema_version"] == "agentic_retrieval_config_v1"
    assert payload["retrieval_mode"] == "agentic"
    assert payload["topics"]["selected_topic_ids"] == ["rag2026-0"]
    assert "OPENROUTER_API_KEY" not in repr(payload)
    assert "PYSERINI_API_TOKEN" not in repr(payload)
