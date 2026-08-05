import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

import pytest

from trec_rag import continuation
from trec_rag.competition_retrieval import _complete, _resume, _retriever_identity
from trec_rag.document_store import DocumentStore
from trec_rag.evaluation import evaluate_ranked, parse_qrels
from trec_rag.evidence import select_top_k_evidence
from trec_rag.generation import generate_placeholder_rag
from trec_rag.pipeline import pipeline_cache_dir, run_pipeline
from trec_rag.pipeline_config import RetrieverConfig, load_pipeline_config
from trec_rag.pipeline_models import (
    QueryVariant,
    RankedCandidate,
    RetrievedCandidate,
)
from trec_rag.query_understanding import build_query_variants
from trec_rag.ranking import coverage_aware_long_doc_rank, passthrough_rank
from trec_rag.remote_pyserini import (
    RemotePyseriniConfig,
    RemotePyseriniThrottled,
    RemoteSearchResponse,
)
from trec_rag.retrieval_cache import (
    DerivationIdentity,
    OrganizerTextNormalizer,
    RetrievalCache,
    RetrievalCacheMiss,
    TransportIdentity,
)
from trec_rag.retrievers import (
    PyseriniRemoteRetriever,
    cache_path,
    normalize_retrieved_candidates,
    request_cache_key,
    transport_identity_for,
)
from trec_rag.topics import Topic


@pytest.fixture(autouse=True)
def _explicit_test_corpus_epoch(monkeypatch):
    monkeypatch.setenv("PYSERINI_CORPUS_EPOCH", "test-epoch")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_config(path, body):
    path.write_text(body, encoding="utf-8")
    return path


def seed_cached_continuation(tmp_path, *, token="continuation-token", query_text="cached query"):
    endpoint = "https://pyserini.test/search"
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )
    query = QueryVariant("topic-a", "original", query_text, "topic")
    cache = RetrievalCache(
        tmp_path,
        DocumentStore(tmp_path / "documents"),
        OrganizerTextNormalizer(),
    )
    identity = TransportIdentity.from_query(
        query_text=query.query_text,
        index_id=config.index,
        endpoint_identity=endpoint,
        corpus_epoch="epoch-1",
        hits=config.hits,
    )
    raw = json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": query.query_text},
            "candidates": [
                {"doc": "cached body", "docid": "doc-a", "rank": 1, "score": 1.0}
            ],
        },
        separators=(",", ":"),
    ).encode()
    cached = cache.commit(
        identity,
        DerivationIdentity.from_normalizer(cache.normalizer),
        query.query_text,
        raw,
    )

    class ConstructionOnlyClient:
        config = RemotePyseriniConfig(endpoint, None, 1, ())

    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        client=ConstructionOnlyClient(),
        retrieval_cache=cache,
        corpus_epoch="epoch-1",
        continuation_ticket=token,
    )
    state = retriever._topic_state(query.topic_id)
    identity_record = retriever._identity_record(query, identity)
    retriever._write_state(
        state / "ticket",
        {
            "ticket": token,
            "not_before_unix": 0,
            "retry_after_seconds": 60,
            "failed_attempt_id": "failed-attempt",
            **identity_record,
        },
    )
    retriever._write_state(
        state / "in-progress",
        {
            "owner": "failed-owner",
            "attempt_id": "failed-attempt",
            "lease_state": "awaiting_continuation",
            "leased_at_unix": time.time(),
            "lease_expires_unix": time.time() + 300,
            **identity_record,
        },
    )
    return retriever, query, identity, cached, state, config


def test_config_loads_defaults_and_rejects_submission_run_id(tmp_path):
    config_path = write_config(
        tmp_path / "config.yaml",
        """
experiment:
  id: rag25_bm25_full_query_v1
submission:
  team_id: local-baseline
topics:
  path: topics.tsv
  format: tsv
query_understanding:
  variants:
    - name: original
      type: original_topic
retrievers:
  - name: climbmix_bm25
    type: pyserini_remote
    query_variants: [original]
    hits: 100
    index: climbmix-400b
ranking:
  type: passthrough
  dedupe:
    by: docid
    keep: best_rank
    preserve_provenance: true
evidence:
  type: top_k
  k: 5
  require_text: true
  allow_fewer: true
generation:
  type: placeholder
evaluation:
  kind: dev_projected_qrels
  qrels: qrels.txt
  metrics: [ndcg@10, recall@100]
  relevance_threshold: 2
""",
    )

    config = load_pipeline_config(config_path)

    assert config.experiment.id == "rag25_bm25_full_query_v1"
    assert config.output_dir == tmp_path / "outputs" / "rag25_bm25_full_query_v1"
    assert config.run_id == "rag25_bm25_full_query_v1"
    assert config.retrievers[0].cache is True

    bad_path = write_config(
        tmp_path / "bad.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline, run_id: old-style}
topics: {path: topics.tsv, format: tsv}
query_understanding: {variants: [{name: original, type: original_topic}]}
retrievers: [{name: climbmix_bm25, type: pyserini_remote, query_variants: [original], hits: 10}]
ranking: {type: passthrough}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
    )

    with pytest.raises(ValueError, match="submission.run_id.*experiment.id"):
        load_pipeline_config(bad_path)


def test_config_allows_disabling_retriever_cache(tmp_path):
    config_path = write_config(
        tmp_path / "config.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: topics.tsv, format: tsv}
query_understanding: {variants: [{name: original, type: original_topic}]}
retrievers:
  - name: climbmix_bm25
    type: pyserini_remote
    query_variants: [original]
    hits: 10
    cache: false
ranking: {type: passthrough}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
    )

    config = load_pipeline_config(config_path)

    assert config.retrievers[0].cache is False


def test_config_loads_coverage_aware_reranker(tmp_path):
    doc_scores = tmp_path / "doc_scores.jsonl"
    chunk_scores = tmp_path / "chunk_scores.jsonl"
    config_path = write_config(
        tmp_path / "config.yaml",
        f"""
experiment: {{id: demo}}
submission: {{team_id: local-baseline}}
topics: {{path: topics.tsv, format: tsv}}
query_understanding: {{variants: [{{name: original, type: original_topic}}]}}
retrievers:
  - name: climbmix_bm25
    type: pyserini_remote
    query_variants: [original]
    hits: 50
ranking:
  type: coverage_aware_long_doc_aggregate
  dedupe:
    by: docid
    keep: best_rank
    preserve_provenance: true
  reranker:
    model: mixedbread-ai/mxbai-rerank-base-v2
    score_source: cached_artifacts
    candidate_depth: 50
    document_score_path: {doc_scores}
    window_score_path: {chunk_scores}
    formula:
      long_document_weight: 0.5
      strongest_passage_weight: 0.5
      coverage_bonus_weight: 0.25
      relative_span_delta: 1.0
      support_cap: 6
      min_new_chars: 800
      top_window_weights: [0.55, 0.25, 0.13, 0.07]
evidence: {{type: top_k, k: 1, allow_fewer: true}}
generation: {{type: placeholder}}
""",
    )

    config = load_pipeline_config(config_path)

    assert config.ranking.type == "coverage_aware_long_doc_aggregate"
    assert config.ranking.reranker is not None
    assert config.ranking.reranker.model == "mixedbread-ai/mxbai-rerank-base-v2"
    assert config.ranking.reranker.candidate_depth == 50
    assert config.ranking.reranker.document_score_path == doc_scores
    assert config.ranking.reranker.window_score_path == chunk_scores
    assert config.ranking.reranker.formula.coverage_bonus_weight == 0.25

    for invalid_depth in ("0", "true", "50.5"):
        invalid_path = write_config(
            tmp_path / f"invalid-depth-{invalid_depth}.yaml",
            config_path.read_text(encoding="utf-8").replace(
                "candidate_depth: 50", f"candidate_depth: {invalid_depth}"
            ),
        )
        with pytest.raises(ValueError, match="candidate_depth"):
            load_pipeline_config(invalid_path)


@pytest.mark.parametrize(
    ("formula_entry", "error"),
    [
        ("top_window_weights: []", "non-empty"),
        ("top_window_weights: [0.0, 1.0]", "positive weight"),
        ("top_window_weights: [-0.1, 1.0]", "finite and non-negative"),
        ("top_window_weights: [.nan]", "finite and non-negative"),
        ("long_document_weight: -0.1", "weights must be non-negative"),
        ("strongest_passage_weight: -0.1", "weights must be non-negative"),
        ("coverage_bonus_weight: -0.1", "weights must be non-negative"),
    ],
)
def test_config_rejects_invalid_coverage_formula_weights(tmp_path, formula_entry, error):
    config_path = write_config(
        tmp_path / "config.yaml",
        f"""
experiment: {{id: demo}}
submission: {{team_id: local-baseline}}
topics: {{path: topics.tsv, format: tsv}}
query_understanding: {{variants: [{{name: original, type: original_topic}}]}}
retrievers: [{{name: bm25, type: pyserini_remote, query_variants: [original], hits: 10}}]
ranking:
  type: coverage_aware_long_doc_aggregate
  reranker:
    model: mixedbread-ai/mxbai-rerank-base-v2
    score_source: cached_artifacts
    document_score_path: document.jsonl
    window_score_path: windows.jsonl
    formula:
      {formula_entry}
evidence: {{type: top_k, k: 1, allow_fewer: true}}
generation: {{type: placeholder}}
""",
    )

    with pytest.raises(ValueError, match=error):
        load_pipeline_config(config_path)


@pytest.mark.parametrize(
    "formula_entry",
    [
        "support_cap: true",
        "support_cap: 1.5",
        "support_cap: 0",
        "min_new_chars: true",
        "min_new_chars: 1.5",
        "min_new_chars: 0",
    ],
)
def test_config_rejects_non_positive_integer_coverage_limits(tmp_path, formula_entry):
    config_path = write_config(
        tmp_path / "config.yaml",
        f"""
experiment: {{id: demo}}
submission: {{team_id: local-baseline}}
topics: {{path: topics.tsv, format: tsv}}
query_understanding: {{variants: [{{name: original, type: original_topic}}]}}
retrievers: [{{name: bm25, type: pyserini_remote, query_variants: [original], hits: 10}}]
ranking:
  type: coverage_aware_long_doc_aggregate
  reranker:
    model: mixedbread-ai/mxbai-rerank-base-v2
    score_source: cached_artifacts
    document_score_path: document.jsonl
    window_score_path: windows.jsonl
    formula:
      {formula_entry}
evidence: {{type: top_k, k: 1, allow_fewer: true}}
generation: {{type: placeholder}}
""",
    )

    with pytest.raises(ValueError, match="must be a positive integer"):
        load_pipeline_config(config_path)


def test_config_rejects_unknown_topic_format(tmp_path):
    config_path = write_config(
        tmp_path / "config.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: topics.data, format: csv}
query_understanding: {variants: [{name: original, type: original_topic}]}
retrievers: [{name: climbmix_bm25, type: pyserini_remote, query_variants: [original], hits: 10}]
ranking: {type: passthrough}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
    )

    with pytest.raises(ValueError, match="topics.format"):
        load_pipeline_config(config_path)


def test_config_rejects_duplicate_names_and_unknown_query_variant(tmp_path):
    duplicate_path = write_config(
        tmp_path / "duplicate.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: topics.tsv, format: tsv}
query_understanding:
  variants:
    - {name: original, type: original_topic}
    - {name: original, type: original_topic}
retrievers:
  - {name: climbmix_bm25, type: pyserini_remote, query_variants: [original], hits: 10}
ranking: {type: passthrough}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
    )
    unknown_path = write_config(
        tmp_path / "unknown.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: topics.tsv, format: tsv}
query_understanding:
  variants:
    - {name: original, type: original_topic}
retrievers:
  - {name: climbmix_bm25, type: pyserini_remote, query_variants: [missing], hits: 10}
ranking: {type: passthrough}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
    )

    with pytest.raises(ValueError, match="duplicate query variant"):
        load_pipeline_config(duplicate_path)
    with pytest.raises(ValueError, match="unknown query variant.*missing"):
        load_pipeline_config(unknown_path)


def test_config_rejects_unsupported_passthrough_dedupe_policy(tmp_path):
    config_path = write_config(
        tmp_path / "bad_dedupe.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: topics.tsv, format: tsv}
query_understanding: {variants: [{name: original, type: original_topic}]}
retrievers: [{name: climbmix_bm25, type: pyserini_remote, query_variants: [original], hits: 10}]
ranking:
  type: passthrough
  dedupe: {by: url, keep: average_score, preserve_provenance: true}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
""",
    )

    with pytest.raises(ValueError, match="dedupe.*docid.*best_rank"):
        load_pipeline_config(config_path)


def test_config_resolves_input_paths_from_shared_checkout_for_linked_worktrees(tmp_path):
    shared = tmp_path / "shared"
    worktree = tmp_path / "worktree"
    git_dir = shared / ".git" / "worktrees" / "wt"
    config_dir = worktree / "configs"
    (shared / "trec-rag-data").mkdir(parents=True)
    worktree.mkdir()
    config_dir.mkdir()
    git_dir.mkdir(parents=True)
    (worktree / "AGENTS.md").write_text("# instructions\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (shared / "trec-rag-data" / "topics.tsv").write_text("31\tPrompt\n", encoding="utf-8")
    (shared / "trec-rag-data" / "qrels.txt").write_text("31 0 doc-a 4\n", encoding="utf-8")
    config_path = write_config(
        config_dir / "config.yaml",
        """
experiment: {id: demo}
submission: {team_id: local-baseline}
topics: {path: trec-rag-data/topics.tsv, format: tsv}
query_understanding: {variants: [{name: original, type: original_topic}]}
retrievers: [{name: climbmix_bm25, type: pyserini_remote, query_variants: [original], hits: 10}]
ranking: {type: passthrough}
evidence: {type: top_k, k: 1, allow_fewer: true}
generation: {type: placeholder}
evaluation:
  qrels: trec-rag-data/qrels.txt
  metrics: [recall@10]
""",
    )

    config = load_pipeline_config(config_path)

    assert config.topics.path == shared / "trec-rag-data" / "topics.tsv"
    assert config.evaluation.qrels == shared / "trec-rag-data" / "qrels.txt"
    assert config.output_dir == worktree / "outputs" / "demo"


def test_original_topic_query_uses_narrative_without_title_concat():
    topic = Topic(
        id="31",
        title="Derived or official title",
        narrative="Full narrative or prompt text.",
    )

    variants = build_query_variants(
        topic,
        variant_configs=[{"name": "original", "type": "original_topic"}],
    )

    assert variants == [
        QueryVariant(
            topic_id="31",
            variant_name="original",
            query_text="Full narrative or prompt text.",
            source_type="original_topic",
        )
    ]


def test_retrieval_normalization_and_cache_path_include_topic_variant_and_retriever():
    query = QueryVariant(
        topic_id="31",
        variant_name="original",
        query_text="e-waste narrative",
        source_type="original_topic",
    )
    response = {
        "candidates": [
            {"rank": 1, "docid": "doc-a", "score": 9.0, "doc": {"contents": "A text"}},
            {"rank": 2, "docid": "doc-b", "score": 8.0, "doc": {"contents": "B text"}},
        ]
    }

    candidates = normalize_retrieved_candidates(
        response,
        query=query,
        retriever_name="climbmix_bm25",
    )

    assert cache_path("31", "original", "climbmix_bm25").name == (
        "31__original__climbmix_bm25.json"
    )
    assert candidates == [
        RetrievedCandidate(
            topic_id="31",
            variant_name="original",
            retriever_name="climbmix_bm25",
            query_text="e-waste narrative",
            docid="doc-a",
            rank=1,
            score=9.0,
            text="A text",
        ),
        RetrievedCandidate(
            topic_id="31",
            variant_name="original",
            retriever_name="climbmix_bm25",
            query_text="e-waste narrative",
            docid="doc-b",
            rank=2,
            score=8.0,
            text="B text",
        ),
    ]


def test_request_cache_key_changes_when_hits_change():
    query = QueryVariant("31", "original", "e-waste narrative", "original_topic")
    first = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=10,
        index="climbmix-400b",
    )
    second = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=100,
        index="climbmix-400b",
    )

    assert request_cache_key(first, query, index_url="https://pyserini.test/search") != (
        request_cache_key(second, query, index_url="https://pyserini.test/search")
    )


def test_request_cache_key_is_full_v2_transport_key_without_pipeline_labels():
    query = QueryVariant("31", "original", "e-waste narrative", "original_topic")
    first = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=10,
        index="climbmix-400b",
    )
    second = RetrieverConfig(
        name="renamed_retriever",
        type="pyserini_remote",
        query_variants=("followup",),
        hits=10,
        index="climbmix-400b",
    )

    first_key = request_cache_key(
        first,
        query,
        index_url="https://pyserini.test/v1/climbmix-400b/search",
        corpus_epoch="epoch-1",
    )
    second_key = request_cache_key(
        second,
        query,
        index_url="https://pyserini.test/v1/climbmix-400b/search",
        corpus_epoch="epoch-1",
    )

    assert len(first_key) == 64
    assert first_key == second_key
    assert first_key != request_cache_key(
        first,
        query,
        index_url="https://pyserini.test/v1/climbmix-400b/search",
        corpus_epoch="epoch-2",
    )


def test_pyserini_remote_retriever_rejects_conflicting_index_url(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_URL", "https://pyserini.test/v1/other-index/search")
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=10,
        index="climbmix-400b",
    )

    with pytest.raises(ValueError, match="INDEX_URL conflicts"):
        PyseriniRemoteRetriever(config, cache_dir=tmp_path)


def test_production_pyserini_construction_requires_explicit_corpus_epoch(tmp_path, monkeypatch):
    monkeypatch.delenv("PYSERINI_CORPUS_EPOCH", raising=False)
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=10,
        index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 10, ())

    with pytest.raises(ValueError, match="corpus_epoch"):
        PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=Client())


def test_transport_builder_rejects_fabricated_index_and_epoch(tmp_path, monkeypatch):
    monkeypatch.delenv("PYSERINI_CORPUS_EPOCH", raising=False)
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=10,
        index="climbmix-400b",
    )
    query = QueryVariant("31", "original", "query", "topic")

    with pytest.raises(ValueError, match="corpus_epoch"):
        transport_identity_for(
            config,
            query,
            index_url="https://pyserini.test/search",
        )
    bad = RetrieverConfig(
        name=config.name,
        type=config.type,
        query_variants=config.query_variants,
        hits=config.hits,
        index="unknown-index",
    )
    with pytest.raises(ValueError, match="index"):
        transport_identity_for(
            bad,
            query,
            index_url="https://pyserini.test/search",
            corpus_epoch="epoch-1",
        )


def test_cached_client_must_expose_search_raw(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )

    class MappingOnlyClient:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

        def search(self, _query):
            return {"api": "v1", "index": "climbmix-400b", "candidates": []}

    with pytest.raises(TypeError, match="search_raw"):
        PyseriniRemoteRetriever(
            config,
            cache_dir=tmp_path,
            client=MappingOnlyClient(),
            corpus_epoch="epoch-1",
        ).retrieve(QueryVariant("31", "original", "query", "topic"))


def test_cache_hit_does_not_construct_live_client_or_rate_limiter(tmp_path, monkeypatch):
    query = QueryVariant("31", "original", "cached query", "topic")
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )
    endpoint = "https://pyserini.test/search"
    identity = TransportIdentity.from_query(
        query_text=query.query_text,
        index_id=config.index,
        endpoint_identity=endpoint,
        corpus_epoch="epoch-1",
        hits=config.hits,
    )
    cache = RetrievalCache(
        tmp_path,
        DocumentStore(tmp_path / "documents"),
        OrganizerTextNormalizer(),
    )
    raw = json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": query.query_text},
            "candidates": [
                {"doc": "cached body", "docid": "doc-a", "rank": 1, "score": 1.0}
            ],
        },
        separators=(",", ":"),
    ).encode()
    cache.commit(identity, DerivationIdentity.from_normalizer(cache.normalizer), query.query_text, raw)

    class ExplodingClient:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("live client construction is not allowed on cache hit")

    monkeypatch.setenv("INDEX_URL", endpoint)
    monkeypatch.setattr("trec_rag.retrievers.RemotePyseriniClient", ExplodingClient)
    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        retrieval_cache=cache,
        corpus_epoch="epoch-1",
    )

    assert [row.docid for row in retriever.retrieve(query)] == ["doc-a"]


def test_cache_hit_continuation_publishes_marker_and_cleans_state_without_hosted_call(
    tmp_path,
    monkeypatch,
):
    seeded, query, identity, cached, state, config = seed_cached_continuation(tmp_path)
    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        retrieval_cache=seeded.retrieval_cache,
        corpus_epoch="epoch-1",
        continuation_ticket="continuation-token",
    )

    class ExplodingClient:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("cache-hit continuation must not construct a hosted client")

    monkeypatch.setattr("trec_rag.retrievers.RemotePyseriniClient", ExplodingClient)

    assert [row.docid for row in retriever.retrieve(query)] == ["doc-a"]
    assert not (state / "ticket").exists()
    assert not (state / "in-progress").exists()
    marker = json.loads((state / "completion").read_text(encoding="utf-8"))
    assert marker["state"] == "completed"
    assert marker["request_key"] == identity.request_key
    assert marker["raw_sha256"] == cached.raw_sha256
    assert retriever.continuation_ticket not in (state / "completion").read_text(
        encoding="utf-8"
    )


def test_cache_hit_continuation_marker_retries_finish_partial_cleanup_idempotently(
    tmp_path,
    monkeypatch,
):
    retriever, query, _identity, _cached, state, config = seed_cached_continuation(tmp_path)
    ticket_path = state / "ticket"
    real_unlink = Path.unlink

    def fail_ticket_unlink(path, *args, **kwargs):
        if path == ticket_path:
            raise OSError("crash at ticket unlink boundary")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_ticket_unlink)
        with pytest.raises(OSError, match="ticket unlink"):
            retriever.retrieve(query)

    assert (state / "completion").exists()
    assert (state / "ticket").exists()
    assert not (state / "in-progress").exists()

    retry = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        retrieval_cache=retriever.retrieval_cache,
        corpus_epoch="epoch-1",
        continuation_ticket="continuation-token",
    )
    assert [row.docid for row in retry.retrieve(query)] == ["doc-a"]
    assert not ticket_path.exists()
    assert (state / "completion").exists()


def test_cache_hit_completion_marker_never_deletes_a_new_active_recovery(
    tmp_path,
    monkeypatch,
):
    retriever, query, _identity, _cached, state, config = seed_cached_continuation(
        tmp_path
    )
    ticket_path = state / "ticket"
    real_unlink = Path.unlink

    def fail_ticket_unlink(path, *args, **kwargs):
        if path == ticket_path:
            raise OSError("crash at ticket unlink boundary")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_ticket_unlink)
        with pytest.raises(OSError, match="ticket unlink"):
            retriever.retrieve(query)

    expected = retriever._identity_record(
        query, retriever._transport_identity(query)
    )
    retriever._write_state(
        state / "in-progress",
        {
            "owner": "new-recovery-owner",
            "attempt_id": "new-recovery-attempt",
            "lease_state": "active_recovery",
            "leased_at_unix": time.time(),
            "lease_expires_unix": time.time() + 300,
            **expected,
        },
    )
    retry = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        retrieval_cache=retriever.retrieval_cache,
        corpus_epoch="epoch-1",
        continuation_ticket="continuation-token",
    )

    with pytest.raises(RuntimeError, match="active active_recovery lease"):
        retry.retrieve(query)

    assert ticket_path.exists()
    assert (state / "in-progress").exists()


def test_cache_hit_completion_marker_allows_two_idempotent_cleanup_retries(
    tmp_path,
    monkeypatch,
):
    retriever, query, _identity, _cached, state, config = seed_cached_continuation(
        tmp_path
    )
    ticket_path = state / "ticket"
    real_unlink = Path.unlink

    def fail_ticket_unlink(path, *args, **kwargs):
        if path == ticket_path:
            raise OSError("crash at ticket unlink boundary")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_ticket_unlink)
        with pytest.raises(OSError, match="ticket unlink"):
            retriever.retrieve(query)

    retries = [
        PyseriniRemoteRetriever(
            config,
            cache_dir=tmp_path,
            retrieval_cache=retriever.retrieval_cache,
            corpus_epoch="epoch-1",
            continuation_ticket="continuation-token",
        )
        for _ in range(2)
    ]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda item: item.retrieve(query), retries))

    assert [[row.docid for row in result] for result in results] == [
        ["doc-a"],
        ["doc-a"],
    ]
    assert not ticket_path.exists()
    assert not (state / "in-progress").exists()


def test_completed_continuation_marker_does_not_block_a_later_fresh_request(
    tmp_path,
):
    retriever, query, _identity, _cached, state, config = seed_cached_continuation(
        tmp_path
    )
    assert [row.docid for row in retriever.retrieve(query)] == ["doc-a"]
    assert (state / "completion").exists()

    fresh = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        client=retriever._client,
        retrieval_cache=retriever.retrieval_cache,
        corpus_epoch="epoch-1",
    )
    fresh_query = QueryVariant("topic-a", "original", "new request", "topic")
    identity = fresh._transport_identity(fresh_query)
    _state, progress_path, attempt_id, _continuation, owner = (
        fresh._reserve_topic_attempt(fresh_query, identity)
    )

    assert not (state / "completion").exists()
    fresh._finish_topic(
        state,
        progress_path,
        owner=owner,
        attempt_id=attempt_id,
    )


def test_cache_hit_continuation_rejects_a_tampered_completion_marker(tmp_path):
    retriever, query, _identity, _cached, state, config = seed_cached_continuation(
        tmp_path
    )
    assert [row.docid for row in retriever.retrieve(query)] == ["doc-a"]
    marker_path = state / "completion"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["raw_sha256"] = "0" * 64
    retriever._write_state(marker_path, marker)

    retry = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        retrieval_cache=retriever.retrieval_cache,
        corpus_epoch="epoch-1",
        continuation_ticket="continuation-token",
    )
    with pytest.raises(RuntimeError, match="completion marker"):
        retry.retrieve(query)


def test_cache_hit_continuation_race_is_serialized_by_topic_state_lock(tmp_path):
    first, query, _identity, _cached, state, _config = seed_cached_continuation(tmp_path)
    second = PyseriniRemoteRetriever(
        first.config,
        cache_dir=tmp_path,
        client=first._client,
        retrieval_cache=first.retrieval_cache,
        corpus_epoch="epoch-1",
        continuation_ticket="continuation-token",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda retriever: retriever.retrieve(query), (first, second))
        )

    assert [[row.docid for row in result] for result in results] == [["doc-a"], ["doc-a"]]
    assert not (state / "ticket").exists()
    assert not (state / "in-progress").exists()
    assert (state / "completion").exists()


def test_cache_hit_continuation_rejects_tampered_completion_marker(tmp_path):
    seeded, query, _identity, _cached, state, config = seed_cached_continuation(tmp_path)
    seeded.retrieve(query)
    marker = json.loads((state / "completion").read_text(encoding="utf-8"))
    marker["raw_sha256"] = "0" * 64
    (state / "completion").write_text(json.dumps(marker), encoding="utf-8")

    retry = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        retrieval_cache=seeded.retrieval_cache,
        corpus_epoch="epoch-1",
        continuation_ticket="continuation-token",
    )
    with pytest.raises(RuntimeError, match="completion marker"):
        retry.retrieve(query)


def test_cache_hit_continuation_marker_write_failure_preserves_unconsumed_state(
    tmp_path,
    monkeypatch,
):
    retriever, query, _identity, _cached, state, _config = seed_cached_continuation(tmp_path)
    original_write_state = retriever._write_state

    def fail_marker_write(path, value):
        if path.name == "completion":
            raise OSError("crash at completion marker boundary")
        return original_write_state(path, value)

    monkeypatch.setattr(retriever, "_write_state", fail_marker_write)
    with pytest.raises(OSError, match="completion marker"):
        retriever.retrieve(query)

    assert not (state / "completion").exists()
    assert (state / "ticket").exists()
    assert (state / "in-progress").exists()


@pytest.mark.parametrize("lease_state", ["active_recovery", "active_transport"])
def test_cache_hit_continuation_rejects_any_unexpired_competing_lease(
    tmp_path,
    lease_state,
):
    retriever, query, _identity, _cached, state, _config = seed_cached_continuation(tmp_path)
    progress = json.loads((state / "in-progress").read_text(encoding="utf-8"))
    progress.update(
        {
            "owner": "another-worker",
            "attempt_id": "competing-attempt",
            "lease_state": lease_state,
            "lease_expires_unix": time.time() + 300,
        }
    )
    retriever._write_state(state / "in-progress", progress)

    with pytest.raises(RuntimeError, match="active .* lease"):
        retriever.retrieve(query)

    assert (state / "ticket").exists()
    assert (state / "in-progress").exists()
    assert not (state / "completion").exists()


def test_cache_hit_continuation_rejects_changed_request_identity(tmp_path):
    retriever, query, _identity, _cached, state, config = seed_cached_continuation(tmp_path)
    changed_query = QueryVariant("topic-a", "original", "changed query", "topic")
    changed_identity = retriever._transport_identity(changed_query)
    raw = json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": changed_query.query_text},
            "candidates": [
                {"doc": "changed body", "docid": "doc-b", "rank": 1, "score": 1.0}
            ],
        },
        separators=(",", ":"),
    ).encode()
    retriever.retrieval_cache.commit(
        changed_identity,
        DerivationIdentity.from_normalizer(retriever.retrieval_cache.normalizer),
        changed_query.query_text,
        raw,
    )
    changed = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        client=retriever._client,
        retrieval_cache=retriever.retrieval_cache,
        corpus_epoch="epoch-1",
        continuation_ticket="continuation-token",
    )

    with pytest.raises(RuntimeError, match="request identity mismatch"):
        changed.retrieve(changed_query)

    assert (state / "ticket").exists()
    assert (state / "in-progress").exists()
    assert not (state / "completion").exists()


def test_cache_hit_continuation_validates_not_before_before_consuming_state(tmp_path):
    retriever, query, _identity, _cached, state, _config = seed_cached_continuation(tmp_path)
    ticket = json.loads((state / "ticket").read_text(encoding="utf-8"))
    ticket["not_before_unix"] = time.time() + 300
    retriever._write_state(state / "ticket", ticket)

    with pytest.raises(RuntimeError, match="not elapsed"):
        retriever.retrieve(query)

    assert (state / "ticket").exists()
    assert (state / "in-progress").exists()
    assert not (state / "completion").exists()


def test_cache_hit_continuation_rejects_unknown_state(tmp_path):
    retriever, query, _identity, _cached, state, _config = seed_cached_continuation(tmp_path)
    progress = json.loads((state / "in-progress").read_text(encoding="utf-8"))
    progress["lease_state"] = "future_state"
    retriever._write_state(state / "in-progress", progress)

    with pytest.raises(RuntimeError, match="unknown"):
        retriever.retrieve(query)

    assert (state / "ticket").exists()
    assert (state / "in-progress").exists()


def test_cache_hit_continuation_rejects_changed_token(tmp_path):
    retriever, query, _identity, _cached, state, config = seed_cached_continuation(tmp_path)
    changed = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        client=retriever._client,
        retrieval_cache=retriever.retrieval_cache,
        corpus_epoch="epoch-1",
        continuation_ticket="changed-token",
    )

    with pytest.raises(RuntimeError, match="continuation ticket"):
        changed.retrieve(query)

    assert (state / "ticket").exists()
    assert (state / "in-progress").exists()
    assert not (state / "completion").exists()


def test_cache_hit_continuation_rejects_consumed_state_without_marker(tmp_path):
    retriever, query, _identity, _cached, state, config = seed_cached_continuation(tmp_path)
    (state / "ticket").unlink()
    (state / "in-progress").unlink()
    consumed = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        client=retriever._client,
        retrieval_cache=retriever.retrieval_cache,
        corpus_epoch="epoch-1",
        continuation_ticket="continuation-token",
    )

    with pytest.raises(RuntimeError, match="invalid or has already been consumed"):
        consumed.retrieve(query)


def test_cache_warm_competition_identity_never_accesses_lazy_client(
    tmp_path,
    monkeypatch,
):
    query = QueryVariant("31", "original", "cached query", "topic")
    endpoint = "https://pyserini.test/search"
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )
    cache = RetrievalCache(
        tmp_path,
        DocumentStore(tmp_path / "documents"),
        OrganizerTextNormalizer(),
    )
    identity = TransportIdentity.from_query(
        query_text=query.query_text,
        index_id="climbmix-400b",
        endpoint_identity=endpoint,
        corpus_epoch="epoch-1",
        hits=1,
    )
    raw = json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": query.query_text},
            "candidates": [
                {"doc": "cached body", "docid": "doc-a", "rank": 1, "score": 1.0}
            ],
        },
        separators=(",", ":"),
    ).encode()
    cache.commit(
        identity,
        DerivationIdentity.from_normalizer(cache.normalizer),
        query.query_text,
        raw,
    )
    monkeypatch.setenv("INDEX_URL", endpoint)
    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        retrieval_cache=cache,
        corpus_epoch="epoch-1",
    )

    def explode(_self):
        raise AssertionError("checkpoint identity must not construct or access the client")

    monkeypatch.setattr(PyseriniRemoteRetriever, "client", property(explode))

    assert _retriever_identity(retriever, retrieval_depth=1) == {
        "name": "climbmix_bm25",
        "type": "pyserini_remote",
        "index": "climbmix-400b",
        "index_url": endpoint,
        "hits": 1,
        "corpus_epoch": "epoch-1",
        "retrieval_cache_schema": "organizer-retrieval-cache-v2",
        "parser_version": "organizer-response-v2",
        "extractor_version": "organizer-exact-doc-string-v1",
        "field_path": ["doc"],
        "scoring_normalizer_version": "whitespace-score-v1",
    }
    assert [candidate.docid for candidate in retriever.retrieve(query)] == ["doc-a"]


def test_lazy_client_cannot_rebind_construction_checkpoint_endpoint(
    tmp_path,
    monkeypatch,
):
    first_endpoint = "https://first.pyserini.test/v1/climbmix-400b/search"
    second_endpoint = "https://second.pyserini.test/v1/climbmix-400b/search"
    monkeypatch.setenv("INDEX_URL", first_endpoint)
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )
    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        corpus_epoch="epoch-1",
    )
    checkpoint_identity = retriever.identity

    class CapturingClient:
        def __init__(self, remote_config):
            self.config = remote_config

    monkeypatch.setenv("INDEX_URL", second_endpoint)
    monkeypatch.setattr("trec_rag.retrievers.RemotePyseriniClient", CapturingClient)

    assert retriever.client.config.index_url == first_endpoint
    assert retriever.identity == checkpoint_identity


def test_pyserini_checkpoint_identity_never_falls_back_to_client():
    class InvalidRetriever:
        config = RetrieverConfig(
            name="climbmix_bm25",
            type="pyserini_remote",
            query_variants=("original",),
            hits=1,
            index="climbmix-400b",
            corpus_epoch="epoch-1",
        )

        @property
        def client(self):
            raise AssertionError("checkpoint identity crossed the lazy client boundary")

    with pytest.raises(ValueError, match="immutable checkpoint identity"):
        _retriever_identity(InvalidRetriever(), retrieval_depth=1)


@pytest.mark.parametrize("changed_identity", ["epoch", "derivation"])
def test_competition_checkpoint_invalidates_epoch_or_derivation_change(
    tmp_path,
    changed_identity,
):
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

    baseline = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path / "baseline",
        client=Client(),
        corpus_epoch="epoch-1",
    )
    changed_cache = RetrievalCache(
        tmp_path / "changed",
        DocumentStore(tmp_path / "documents"),
        OrganizerTextNormalizer(version="organizer-exact-doc-string-v2"),
    )
    changed = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path / "changed",
        client=Client(),
        retrieval_cache=(changed_cache if changed_identity == "derivation" else None),
        corpus_epoch=("epoch-2" if changed_identity == "epoch" else "epoch-1"),
    )
    baseline_identity = _retriever_identity(baseline, retrieval_depth=1)
    changed_checkpoint_identity = _retriever_identity(changed, retrieval_depth=1)
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "audit.json").write_text("{}", encoding="utf-8")
    manifest = root / "complete.json"
    baseline_expected = {
        "schema_version": "test",
        "phase": "retrieve",
        "retriever": baseline_identity,
    }
    _complete(manifest, root, baseline_expected, ("audit.json",))

    assert _resume(manifest, root, baseline_expected, ("audit.json",)) is True
    with pytest.raises(ValueError, match="checkpoint input identity changed"):
        _resume(
            manifest,
            root,
            {**baseline_expected, "retriever": changed_checkpoint_identity},
            ("audit.json",),
        )


def test_offline_miss_fails_before_live_client_construction(tmp_path, monkeypatch):
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )

    class ExplodingClient:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("offline miss crossed the external boundary")

    monkeypatch.setenv("INDEX_URL", "https://pyserini.test/search")
    monkeypatch.setattr("trec_rag.retrievers.RemotePyseriniClient", ExplodingClient)
    with pytest.raises(RetrievalCacheMiss, match="offline cache miss"):
        PyseriniRemoteRetriever(
            config,
            cache_dir=tmp_path,
            corpus_epoch="epoch-1",
            offline=True,
        ).retrieve(QueryVariant("31", "original", "query", "topic"))


def test_arbitrary_retrieval_namespace_uses_shared_document_store(tmp_path):
    namespace = tmp_path / "cache" / "retrieval" / "facet_2025_v2"
    config = RetrieverConfig(
        name="facet",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=namespace,
        client=Client(),
        corpus_epoch="epoch-1",
    )

    assert retriever.retrieval_cache.document_store._root == (
        tmp_path / "cache" / "documents" / "v1"
    )


def test_pyserini_remote_retriever_uses_matching_index_url(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_URL", "https://pyserini.test/v1/climbmix-400b/search")
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=10,
        index="climbmix-400b",
    )

    retriever = PyseriniRemoteRetriever(config, cache_dir=tmp_path)

    assert retriever.client.config.index_url == "https://pyserini.test/v1/climbmix-400b/search"


def test_pyserini_remote_retriever_reads_cached_response_when_cache_enabled(tmp_path):
    query = QueryVariant("31", "original", "e-waste narrative", "original_topic")
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=10,
        index="climbmix-400b",
    )

    class FailingClient:
        config = RemotePyseriniConfig(
            index_url="https://pyserini.test/search",
            api_token=None,
            hits=10,
            queries=(),
        )

        def search(self, _query):
            raise AssertionError("cache hit should not call remote search")

    raw = json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": query.query_text},
            "candidates": [
                {"rank": 1, "docid": "doc-cached", "score": 7.0, "doc": "Cached"}
            ]
        },
        separators=(",", ":"),
    ).encode()
    identity = TransportIdentity.from_query(
        query_text=query.query_text,
        index_id=config.index,
        endpoint_identity=FailingClient.config.index_url,
        corpus_epoch="test-epoch",
        hits=config.hits,
    )
    cache = RetrievalCache(tmp_path, DocumentStore(tmp_path / "documents"), OrganizerTextNormalizer())
    cache.commit(
        identity,
        DerivationIdentity.from_normalizer(cache.normalizer),
        query.query_text,
        raw,
    )

    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        client=FailingClient(),
        retrieval_cache=cache,
    )

    candidates = retriever.retrieve(query)

    assert [candidate.docid for candidate in candidates] == ["doc-cached"]
    assert retriever.cache_summary()["hits"] == 1
    assert retriever.cache_summary()["misses"] == 0


def test_pyserini_continuation_reuses_verified_success_and_sends_only_missing(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25", type="pyserini_remote", query_variants=("original",),
        hits=10, index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 10, ())
        def __init__(self): self.calls = []
        def search_raw(self, query_text, *, raw_sink=None):
            self.calls.append(query_text)
            raw = json.dumps(
                {
                    "api": "v1",
                    "index": "climbmix-400b",
                    "query": {"text": query_text},
                    "candidates": [
                        {"rank": 1, "docid": query_text, "score": 1, "doc": query_text}
                    ],
                }
            ).encode()
            if raw_sink: raw_sink(raw)
            return RemoteSearchResponse(raw, json.loads(raw), hashlib.sha256(raw).hexdigest())

    client = Client()
    first = PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client)
    first.retrieve(QueryVariant("1", "original", "already done", "topic"))

    continuation = PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client)
    continuation.retrieve(QueryVariant("1", "original", "already done", "topic"))
    continuation.retrieve(QueryVariant("2", "original", "still missing", "topic"))

    assert client.calls == ["already done", "still missing"]
    assert continuation.cache_stats.hits == 1
    assert continuation.cache_stats.misses == 1


def test_pyserini_429_requires_ticket_and_reserves_ledger_before_transport(tmp_path):
    query = QueryVariant("15", "original", "throttled query", "topic")
    config = RetrieverConfig(
        name="climbmix_bm25", type="pyserini_remote", query_variants=("original",),
        hits=10, index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 10, ())
        def __init__(self): self.throttle = True
        def search_raw(self, query_text, *, raw_sink):
            ledger = tmp_path / "topic-state" / "15" / "ledger"
            assert json.loads(ledger.read_text().splitlines()[-1])["event"] == "reserved"
            if self.throttle:
                raw_sink(b'{"error":"slow down"}')
                raise RemotePyseriniThrottled(0)
            raw = json.dumps(
                {
                    "api": "v1",
                    "index": "climbmix-400b",
                    "query": {"text": query_text},
                    "candidates": [],
                },
                separators=(",", ":"),
            ).encode()
            raw_sink(raw)
            return RemoteSearchResponse(raw, json.loads(raw), hashlib.sha256(raw).hexdigest())

    client = Client()
    with pytest.raises(RemotePyseriniThrottled) as exc:
        PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client).retrieve(query)
    ticket = exc.value.continuation_ticket
    assert list((tmp_path / "v2" / "attempts").rglob("response.bin"))

    client.throttle = False
    with pytest.raises(RuntimeError, match="explicit continuation"):
        PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client).retrieve(query)
    PyseriniRemoteRetriever(
        config, cache_dir=tmp_path, client=client, continuation_ticket=ticket
    ).retrieve(query)
    assert not (tmp_path / "topic-state" / "15" / "ticket").exists()
    completion = tmp_path / "topic-state" / "15" / "completion"
    assert completion.exists()
    assert ticket not in completion.read_text(encoding="utf-8")
    ledger = [
        json.loads(line)
        for line in (tmp_path / "topic-state" / "15" / "ledger").read_text().splitlines()
    ]
    reserved_ids = [row["attempt_id"] for row in ledger if row["event"] == "reserved"]
    # A short back-off is retried in place first, and each retry is separately
    # reserved, so the ticket names the attempt that actually gave up.
    assert any(row.get("retry_index") for row in ledger)
    continuation_record = next(row for row in ledger if "continuation_of" in row)
    assert continuation_record["continuation_of"] in reserved_ids
    assert continuation_record["continuation_ticket_sha256"] == hashlib.sha256(ticket.encode()).hexdigest()
    assert ticket not in json.dumps(ledger)
    with pytest.raises(RuntimeError, match="invalid or has already been consumed"):
        PyseriniRemoteRetriever(
            config, cache_dir=tmp_path, client=client, continuation_ticket=ticket
        ).retrieve(QueryVariant("16", "original", "another query", "topic"))


def test_only_throttling_latches_the_shared_retrieval_budget(tmp_path):
    """A transport error fails its own request; it must not block later ones.

    Latching every query behind a ticket bound to one query wedged the retriever
    until somebody replayed a query they had no way to know.
    """
    query = QueryVariant("15", "original", "query", "topic")
    other = QueryVariant("15", "original", "a different query", "topic")
    config = RetrieverConfig(
        name="bm25", type="pyserini_remote", query_variants=("original",),
        hits=10, index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 10, ())
        mode = "error"
        def search_raw(self, query_text, *, raw_sink):
            if self.mode == "error":
                raw_sink(b'{"error":"transport failed"}')
                raise ConnectionError("transient network failure")
            raw = json.dumps(
                {
                    "api": "v1",
                    "index": "climbmix-400b",
                    "query": {"text": query_text},
                    "candidates": [],
                },
                separators=(",", ":"),
            ).encode()
            raw_sink(raw)
            return RemoteSearchResponse(
                raw,
                json.loads(raw),
                hashlib.sha256(raw).hexdigest(),
            )

    client = Client()
    with pytest.raises(ConnectionError):
        PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client).retrieve(query)

    assert not (tmp_path / "topic-state" / "15" / "ticket").exists()
    assert not (tmp_path / "topic-state" / "15" / "in-progress").exists()

    # An unrelated query still works: the failure did not latch the retriever.
    client.mode = "success"
    PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client).retrieve(other)

    ledger = (tmp_path / "topic-state" / "15" / "ledger").read_text(encoding="utf-8")
    assert '"event": "failed"' in ledger, "the failure is still recorded"
    assert "ConnectionError" in ledger


def test_failed_continuation_mints_new_explicit_recovery_ticket(tmp_path):
    query = QueryVariant("15", "original", "query", "topic")
    config = RetrieverConfig(
        name="bm25", type="pyserini_remote", query_variants=("original",),
        hits=10, index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 10, ())
        mode = "throttle"
        def search_raw(self, query_text, *, raw_sink):
            if self.mode == "throttle":
                raw_sink(b'{"error":"slow down"}')
                raise RemotePyseriniThrottled(0)
            if self.mode == "error":
                raw_sink(b'{"error":"malformed response"}')
                raise ValueError("malformed response")
            raw = json.dumps(
                {
                    "api": "v1",
                    "index": "climbmix-400b",
                    "query": {"text": query_text},
                    "candidates": [],
                },
                separators=(",", ":"),
            ).encode()
            raw_sink(raw)
            return RemoteSearchResponse(raw, json.loads(raw), hashlib.sha256(raw).hexdigest())

    client = Client()
    with pytest.raises(RemotePyseriniThrottled) as first:
        PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client).retrieve(query)
    client.mode = "error"
    with pytest.raises(ValueError):
        PyseriniRemoteRetriever(
            config, cache_dir=tmp_path, client=client,
            continuation_ticket=first.value.continuation_ticket,
        ).retrieve(query)
    # A failed continuation clears the latch rather than minting another
    # ticket, so a transport blip cannot leave the retriever wedged.
    assert not (tmp_path / "topic-state" / "15" / "in-progress").exists()
    assert not (tmp_path / "topic-state" / "15" / "ticket").exists()

    client.mode = "success"
    PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client).retrieve(query)


def test_pyserini_remote_retriever_cache_false_bypasses_response_cache(tmp_path):
    query = QueryVariant("31", "original", "e-waste narrative", "original_topic")
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=10,
        index="climbmix-400b",
        cache=False,
    )

    class CountingClient:
        config = RemotePyseriniConfig(
            index_url="https://pyserini.test/search",
            api_token=None,
            hits=10,
            queries=(),
        )

        def __init__(self):
            self.calls = []

        def search(self, query_text):
            self.calls.append(query_text)
            return {
                "candidates": [
                    {"rank": 1, "docid": "doc-fresh", "score": 9.0, "doc": {"text": "Fresh"}}
                ]
            }

    stale_cache = tmp_path / cache_path(
        query.topic_id,
        query.variant_name,
        config.name,
        request_cache_key(config, query, index_url=CountingClient.config.index_url),
    )
    stale_cache.write_text(
        json.dumps(
            {
                "response": {
                    "candidates": [
                        {"rank": 1, "docid": "doc-stale", "score": 1.0, "doc": {"text": "Stale"}}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    original_cache_text = stale_cache.read_text(encoding="utf-8")
    client = CountingClient()
    retriever = PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client)

    candidates = retriever.retrieve(query)

    assert client.calls == ["e-waste narrative"]
    assert [candidate.docid for candidate in candidates] == ["doc-fresh"]
    assert stale_cache.read_text(encoding="utf-8") == original_cache_text
    assert not stale_cache.with_suffix(".meta.json").exists()
    assert (tmp_path / "topic-state" / "31" / "ledger").exists()
    assert retriever.cache_summary()["bypasses"] == 1


def test_concurrent_same_topic_cache_miss_enters_remote_once(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25", type="pyserini_remote", query_variants=("original",),
        hits=1, index="climbmix-400b",
    )
    query = QueryVariant("topic-a", "original", "same query", "topic")

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

        def __init__(self):
            self.calls = 0

        def search_raw(self, query_text, *, raw_sink):
            self.calls += 1
            raw = json.dumps(
                {
                    "api": "v1",
                    "index": "climbmix-400b",
                    "query": {"text": query_text},
                    "candidates": [
                        {"rank": 1, "docid": "doc-a", "score": 1.0, "doc": "A"}
                    ],
                }
            ).encode()
            raw_sink(raw)
            return RemoteSearchResponse(raw, json.loads(raw), hashlib.sha256(raw).hexdigest())

    client = Client()
    retrievers = [
        PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client),
        PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client),
    ]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [
            future.result(timeout=5)
            for future in [executor.submit(retriever.retrieve, query) for retriever in retrievers]
        ]

    assert client.calls == 1
    assert [[candidate.docid for candidate in result] for result in results] == [["doc-a"], ["doc-a"]]


def test_remote_success_seals_and_publishes_the_same_attempt(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )
    query = QueryVariant("topic-a", "original", "same attempt", "topic")

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

        def search_raw(self, query_text, *, raw_sink):
            raw = json.dumps(
                {
                    "api": "v1",
                    "index": "climbmix-400b",
                    "query": {"text": query_text},
                    "candidates": [
                        {"doc": "A", "docid": "doc-a", "rank": 1, "score": 1.0}
                    ],
                },
                separators=(",", ":"),
            ).encode()
            raw_sink(raw)
            return RemoteSearchResponse(raw, json.loads(raw), hashlib.sha256(raw).hexdigest())

    retriever = PyseriniRemoteRetriever(
        config, cache_dir=tmp_path, client=Client(), corpus_epoch="epoch-1"
    )
    retriever.retrieve(query)

    identity = retriever._transport_identity(query)
    attempts = [
        path
        for path in (retriever.retrieval_cache.v2_root / "attempts" / identity.request_key).iterdir()
        if path.is_dir()
    ]
    assert len(attempts) == 1
    assert (attempts[0] / "manifest.json").exists()
    ledger = (tmp_path / "topic-state" / "topic-a" / "ledger").read_text()
    assert attempts[0].name in ledger
    assert "attempt_manifest" in ledger


def test_late_old_worker_completion_cannot_clear_takeover_state(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

    retriever = PyseriniRemoteRetriever(
        config, cache_dir=tmp_path, client=Client(), corpus_epoch="epoch-1"
    )
    state = tmp_path / "topic-state" / "topic-a"
    state.mkdir(parents=True)
    progress = state / "in-progress"
    progress.write_text(
        json.dumps({"owner": "new-owner", "attempt_id": "new-attempt"}),
        encoding="utf-8",
    )
    ticket = state / "ticket"
    ticket.write_text(json.dumps({"ticket": "takeover-ticket"}), encoding="utf-8")

    retriever._finish_topic(
        state,
        progress,
        owner="old-owner",
        attempt_id="old-attempt",
    )

    assert progress.exists()
    assert ticket.exists()


def test_wrong_query_continuation_token_cannot_delete_live_lease(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        client=Client(),
        corpus_epoch="epoch-1",
        continuation_ticket="recovery-token",
    )
    original_query = QueryVariant("topic-a", "original", "original query", "topic")
    wrong_query = QueryVariant("topic-a", "original", "wrong query", "topic")
    original_identity = retriever._transport_identity(original_query)
    state = retriever._topic_state("topic-a")
    ticket = {
        "ticket": "recovery-token",
        "not_before_unix": 0,
        "retry_after_seconds": 60,
        "failed_attempt_id": "failed-attempt",
        **retriever._identity_record(original_query, original_identity),
    }
    progress = {
        "owner": "live-owner",
        "attempt_id": "failed-attempt",
        "lease_state": "awaiting_continuation",
        "leased_at_unix": time.time(),
        "lease_expires_unix": time.time() + 300,
        **retriever._identity_record(original_query, original_identity),
    }
    retriever._write_state(state / "ticket", ticket)
    retriever._write_state(state / "in-progress", progress)
    ticket_before = (state / "ticket").read_bytes()
    progress_before = (state / "in-progress").read_bytes()

    with pytest.raises(RuntimeError, match="request identity mismatch"):
        retriever._reserve_topic_attempt(
            wrong_query,
            retriever._transport_identity(wrong_query),
        )

    assert (state / "ticket").read_bytes() == ticket_before
    assert (state / "in-progress").read_bytes() == progress_before


def test_same_token_cannot_take_over_active_recovery_owner(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        client=Client(),
        corpus_epoch="epoch-1",
        continuation_ticket="recovery-token",
    )
    query = QueryVariant("topic-a", "original", "original query", "topic")
    identity = retriever._transport_identity(query)
    state = retriever._topic_state("topic-a")
    identity_record = retriever._identity_record(query, identity)
    retriever._write_state(
        state / "ticket",
        {
            "ticket": "recovery-token",
            "not_before_unix": 0,
            "retry_after_seconds": 60,
            "failed_attempt_id": "failed-attempt",
            **identity_record,
        },
    )
    retriever._write_state(
        state / "in-progress",
        {
            "owner": "active-recovery-owner",
            "attempt_id": "active-recovery-attempt",
            "lease_state": "active_recovery",
            "leased_at_unix": time.time(),
            "lease_expires_unix": time.time() + 300,
            **identity_record,
        },
    )
    progress_before = (state / "in-progress").read_bytes()

    with pytest.raises(RuntimeError, match="active recovery lease"):
        retriever._reserve_topic_attempt(query, identity)

    assert (state / "in-progress").read_bytes() == progress_before


def test_awaiting_continuation_transitions_to_active_recovery(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        client=Client(),
        corpus_epoch="epoch-1",
        continuation_ticket="recovery-token",
    )
    query = QueryVariant("topic-a", "original", "original query", "topic")
    identity = retriever._transport_identity(query)
    state = retriever._topic_state("topic-a")
    identity_record = retriever._identity_record(query, identity)
    retriever._write_state(
        state / "ticket",
        {
            "ticket": "recovery-token",
            "not_before_unix": 0,
            "retry_after_seconds": 60,
            "failed_attempt_id": "failed-attempt",
            **identity_record,
        },
    )
    retriever._write_state(
        state / "in-progress",
        {
            "owner": "failed-owner",
            "attempt_id": "failed-attempt",
            "lease_state": "awaiting_continuation",
            "leased_at_unix": time.time(),
            "lease_expires_unix": time.time() + 300,
            **identity_record,
        },
    )

    _state, progress_path, attempt_id, _continuation, owner = (
        retriever._reserve_topic_attempt(query, identity)
    )
    active = json.loads(progress_path.read_text(encoding="utf-8"))

    assert active["lease_state"] == "active_recovery"
    assert active["attempt_id"] == attempt_id
    assert active["owner"] == owner


@pytest.mark.parametrize("slow_phase", ["transport", "seal", "publication"])
def test_topic_lease_heartbeats_through_slow_retrieval_phases(
    tmp_path,
    monkeypatch,
    slow_phase,
):
    monkeypatch.setattr("trec_rag.retrievers.TOPIC_LEASE_SECONDS", 0.12)
    entered = Event()
    release = Event()
    query = QueryVariant("topic-a", "original", "slow query", "topic")
    raw = json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": query.query_text},
            "candidates": [
                {"doc": "A", "docid": "doc-a", "rank": 1, "score": 1.0}
            ],
        },
        separators=(",", ":"),
    ).encode()
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

        def search_raw(self, _query, *, raw_sink):
            if slow_phase == "transport":
                entered.set()
                if not release.wait(timeout=3):
                    raise AssertionError("transport test gate was not released")
            raw_sink(raw)
            return RemoteSearchResponse(
                raw,
                json.loads(raw),
                hashlib.sha256(raw).hexdigest(),
            )

    retriever = PyseriniRemoteRetriever(
        config,
        cache_dir=tmp_path,
        client=Client(),
        corpus_epoch="epoch-1",
    )
    if slow_phase == "seal":
        original_seal = retriever.retrieval_cache.seal_attempt

        def slow_seal(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise AssertionError("seal test gate was not released")
            return original_seal(*args, **kwargs)

        monkeypatch.setattr(retriever.retrieval_cache, "seal_attempt", slow_seal)
    elif slow_phase == "publication":
        original_commit = retriever.retrieval_cache.commit

        def slow_commit(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise AssertionError("publication test gate was not released")
            return original_commit(*args, **kwargs)

        monkeypatch.setattr(retriever.retrieval_cache, "commit", slow_commit)

    failures: list[BaseException] = []

    def retrieve() -> None:
        try:
            retriever.retrieve(query)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = Thread(target=retrieve)
    worker.start()
    assert entered.wait(timeout=2)
    progress_path = tmp_path / "topic-state" / "topic-a" / "in-progress"
    renewed = None
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            current = json.loads(progress_path.read_text(encoding="utf-8"))
            if "renewed_at_unix" in current:
                renewed = current
                break
            time.sleep(0.02)
        assert renewed is not None
        assert renewed["lease_expires_unix"] > time.time()
    finally:
        release.set()
        worker.join(timeout=3)

    assert not worker.is_alive()
    assert failures == []
    assert not progress_path.exists()


def test_topic_scoped_throttle_does_not_block_another_topic(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25", type="pyserini_remote", query_variants=("original",),
        hits=1, index="climbmix-400b",
    )

    class Throttled:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

        def search_raw(self, _query, *, raw_sink):
            raw_sink(b'{"error":"slow"}')
            raise RemotePyseriniThrottled(600)

    class Successful:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

        def search_raw(self, query_text, *, raw_sink):
            raw = json.dumps(
                {
                    "api": "v1",
                    "index": "climbmix-400b",
                    "query": {"text": query_text},
                    "candidates": [
                        {"rank": 1, "docid": "b", "score": 1, "doc": "B"}
                    ],
                },
                separators=(",", ":"),
            ).encode()
            raw_sink(raw)
            return RemoteSearchResponse(raw, json.loads(raw), hashlib.sha256(raw).hexdigest())

    with pytest.raises(RemotePyseriniThrottled):
        PyseriniRemoteRetriever(
            config, cache_dir=tmp_path, client=Throttled()
        ).retrieve(QueryVariant("topic-a", "original", "A", "topic"))

    result = PyseriniRemoteRetriever(
        config, cache_dir=tmp_path, client=Successful()
    ).retrieve(QueryVariant("topic-b", "original", "B", "topic"))

    assert [candidate.docid for candidate in result] == ["b"]
    assert (tmp_path / "topic-state" / "topic-a" / "ticket").exists()
    assert not (tmp_path / "topic-state" / "topic-b" / "ticket").exists()


def test_expired_topic_lease_is_taken_over_and_success_cleans_state(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25", type="pyserini_remote", query_variants=("original",),
        hits=1, index="climbmix-400b",
    )
    query = QueryVariant("topic-a", "original", "A", "topic")

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

        def search_raw(self, query_text, *, raw_sink):
            raw = json.dumps(
                {
                    "api": "v1",
                    "index": "climbmix-400b",
                    "query": {"text": query_text},
                    "candidates": [
                        {"rank": 1, "docid": "a", "score": 1, "doc": "A"}
                    ],
                },
                separators=(",", ":"),
            ).encode()
            raw_sink(raw)
            return RemoteSearchResponse(raw, json.loads(raw), hashlib.sha256(raw).hexdigest())

    retriever = PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=Client())
    identity = retriever._transport_identity(query)
    state = tmp_path / "topic-state" / "topic-a"
    state.mkdir(parents=True)
    (state / "in-progress").write_text(
        json.dumps(
                {
                    "owner": "dead-worker",
                    "attempt_id": "dead-attempt",
                    "lease_state": "active_transport",
                    "lease_expires_unix": 0,
                **retriever._identity_record(query, identity),
            }
        ),
        encoding="utf-8",
    )

    retriever.retrieve(query)

    assert not (state / "in-progress").exists()
    assert not (state / "ticket").exists()


def test_throttle_renews_topic_lease_until_explicit_recovery(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25", type="pyserini_remote", query_variants=("original",),
        hits=1, index="climbmix-400b",
    )
    observed = {}

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

        def search_raw(self, _query, *, raw_sink):
            raw_sink(b'{"error":"slow"}')
            state = tmp_path / "topic-state" / "topic-a" / "in-progress"
            observed.update(json.loads(state.read_text(encoding="utf-8")))
            raise RemotePyseriniThrottled(600)

    with pytest.raises(RemotePyseriniThrottled):
        PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=Client()).retrieve(
            QueryVariant("topic-a", "original", "A", "topic")
        )

    assert observed["lease_expires_unix"] > observed["leased_at_unix"]
    assert observed["owner"]
    assert observed["attempt_id"]


def test_continuation_cli_can_inspect_and_discard_one_topic_shard(tmp_path):
    state = tmp_path / "topic-state" / "topic-a"
    state.mkdir(parents=True)
    (state / "ticket").write_text(
        json.dumps(
            {
                "ticket": "topic-ticket",
                "not_before_unix": 0,
                "retry_after_seconds": 1,
                "failed_attempt_id": "attempt-a",
                "topic_id": "topic-a",
                "query": "A",
            }
        ),
        encoding="utf-8",
    )

    report = continuation.describe(tmp_path, topic_id="topic-a")
    assert report["state"] == "pending"
    assert report["topic_id"] == "topic-a"
    assert continuation.discard(tmp_path, topic_id="topic-a")["state"] == "clear"


def test_offline_retriever_miss_never_reserves_topic_or_calls_client(tmp_path):
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1,
        index="climbmix-400b",
    )

    class FailingClient:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 1, ())

        def search_raw(self, *_args, **_kwargs):
            raise AssertionError("offline lookup must not reach the client")

    with pytest.raises(RetrievalCacheMiss, match="offline cache miss"):
        PyseriniRemoteRetriever(
            config,
            cache_dir=tmp_path,
            client=FailingClient(),
            offline=True,
        ).retrieve(QueryVariant("topic-a", "original", "A", "topic"))
    assert not (tmp_path / "topic-state").exists()


def test_passthrough_ranking_dedupes_docids_and_preserves_provenance():
    candidates = [
        RetrievedCandidate("31", "original", "bm25", "query", "doc-b", 2, 8.0, "B"),
        RetrievedCandidate("31", "original", "bm25", "query", "doc-a", 1, 9.0, "A"),
        RetrievedCandidate("31", "original", "bm25", "query", "doc-a", 3, 7.0, "A later"),
    ]

    ranked = passthrough_rank(candidates)

    assert [(row.docid, row.rank, row.score) for row in ranked] == [
        ("doc-a", 1, 9.0),
        ("doc-b", 2, 8.0),
    ]
    assert len(ranked[0].provenance) == 2
    assert {item["source_rank"] for item in ranked[0].provenance} == {1, 3}


def test_passthrough_ranking_restarts_final_ranks_per_topic():
    candidates = [
        RetrievedCandidate("31", "original", "bm25", "query", "doc-a", 1, 9.0, "A"),
        RetrievedCandidate("32", "original", "bm25", "query", "doc-b", 1, 8.0, "B"),
    ]

    ranked = passthrough_rank(candidates)

    assert [(row.topic_id, row.docid, row.rank) for row in ranked] == [
        ("31", "doc-a", 1),
        ("32", "doc-b", 1),
    ]


def test_coverage_aware_ranking_rejects_non_finite_cached_scores(tmp_path):
    candidates = [
        RetrievedCandidate("31", "original", "bm25", "query", "doc-a", 1, 1.0, "A")
    ]
    document_scores = tmp_path / "document.jsonl"
    document_scores.write_text(
        json.dumps({"topic_id": "31", "docid": "doc-a", "score": float("nan")}) + "\n",
        encoding="utf-8",
    )
    window_scores = tmp_path / "window.jsonl"
    window_scores.write_text(
        json.dumps(
            {
                "topic_id": "31",
                "docid": "doc-a",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 1,
                "score": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="document score must be finite"):
        coverage_aware_long_doc_rank(
            candidates,
            document_score_path=document_scores,
            window_score_path=window_scores,
            candidate_depth=1,
            long_document_weight=0.5,
            strongest_passage_weight=0.5,
            coverage_bonus_weight=0.25,
            relative_span_delta=1.0,
            support_cap=6,
            min_new_chars=1,
            top_window_weights=(1.0,),
        )


def test_passthrough_rejects_multiple_retrieval_streams():
    candidates = [
        RetrievedCandidate("31", "original", "bm25", "query", "doc-a", 1, 9.0, "A"),
        RetrievedCandidate("31", "rewrite", "bm25", "query", "doc-b", 1, 8.0, "B"),
    ]

    with pytest.raises(ValueError, match="passthrough.*one retrieval stream"):
        passthrough_rank(candidates)


def test_evidence_allows_fewer_text_bearing_candidates_and_placeholder_rag_is_cited():
    topic = Topic(id="31", title="E-waste", narrative="Explain e-waste impacts.")
    ranked = [
        RankedCandidate(
            topic_id="31",
            docid="doc-a",
            rank=1,
            score=9.0,
            text="A useful evidence sentence.",
            provenance=[],
        ),
        RankedCandidate(
            topic_id="31",
            docid="doc-b",
            rank=2,
            score=8.0,
            text="",
            provenance=[],
        ),
    ]

    evidence = select_top_k_evidence(ranked, k=2, require_text=True, allow_fewer=True)
    record = generate_placeholder_rag(
        topic,
        evidence,
        team_id="local-baseline",
        run_id="rag25_bm25_full_query_v1",
    )

    assert [item.docid for item in evidence] == ["doc-a"]
    assert record["references"] == ["doc-a"]
    assert record["answer"][0]["citations"] == [0]
    assert "placeholder" in record["answer"][0]["text"].lower()


def test_qrels_metrics_handle_graded_labels_and_unjudged_docs(tmp_path):
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text(
        "31 0 doc-a 4\n"
        "31 0 doc-b 2\n"
        "31 0 doc-c 1\n"
        "32 0 doc-z 0\n",
        encoding="utf-8",
    )
    ranked = [
        RankedCandidate("31", "doc-unjudged", 1, 10.0, "", []),
        RankedCandidate("31", "doc-b", 2, 9.0, "", []),
        RankedCandidate("31", "doc-a", 3, 8.0, "", []),
        RankedCandidate("32", "doc-z", 1, 5.0, "", []),
    ]

    qrels = parse_qrels(qrels_path)
    metrics = evaluate_ranked(
        ranked,
        qrels,
        metric_names=["ndcg@3", "recall@3"],
        relevance_threshold=2,
    )

    assert metrics["per_topic"]["31"]["recall@3"] == 1.0
    assert 0.0 < metrics["per_topic"]["31"]["ndcg@3"] < 1.0
    assert metrics["per_topic"]["32"]["recall@3"] == 0.0
    assert metrics["metrics"]["recall@3"] == 0.5


def test_qrels_metrics_report_graded_recall_and_ideal_dcg_coverage(tmp_path):
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text(
        "31 0 doc-a 4\n"
        "31 0 doc-b 2\n"
        "31 0 doc-c 1\n"
        "31 0 doc-d 0\n",
        encoding="utf-8",
    )
    ranked = [
        RankedCandidate("31", "doc-unjudged", 1, 10.0, "", []),
        RankedCandidate("31", "doc-b", 2, 9.0, "", []),
        RankedCandidate("31", "doc-c", 3, 8.0, "", []),
    ]

    qrels = parse_qrels(qrels_path)
    metrics = evaluate_ranked(
        ranked,
        qrels,
        metric_names=["graded_recall@3", "ideal_dcg_coverage@3"],
        relevance_threshold=2,
    )

    assert metrics["per_topic"]["31"]["graded_recall@3"] == pytest.approx(3 / 7)
    expected_candidate_idcg = 3 / math.log2(2) + 1 / math.log2(3)
    expected_true_idcg = 15 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert metrics["per_topic"]["31"]["ideal_dcg_coverage@3"] == pytest.approx(
        expected_candidate_idcg / expected_true_idcg
    )


def test_qrels_metrics_report_precision_hit_rate_and_relevant_count(tmp_path):
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text(
        "31 0 doc-a 4\n"
        "31 0 doc-b 2\n"
        "31 0 doc-c 1\n"
        "32 0 doc-z 2\n",
        encoding="utf-8",
    )
    ranked = [
        RankedCandidate("31", "doc-unjudged", 1, 10.0, "", []),
        RankedCandidate("31", "doc-b", 2, 9.0, "", []),
        RankedCandidate("31", "doc-c", 3, 8.0, "", []),
        RankedCandidate("32", "doc-unjudged", 1, 5.0, "", []),
    ]

    metrics = evaluate_ranked(
        ranked,
        parse_qrels(qrels_path),
        metric_names=[
            "precision@3",
            "hit_rate@3",
            "relevant_count@3",
            "judged_count@3",
            "judged_rate@3",
        ],
        relevance_threshold=2,
    )

    assert metrics["per_topic"]["31"]["precision@3"] == pytest.approx(1 / 3)
    assert metrics["per_topic"]["31"]["hit_rate@3"] == 1.0
    assert metrics["per_topic"]["31"]["relevant_count@3"] == 1
    assert metrics["per_topic"]["32"]["precision@3"] == 0.0
    assert metrics["per_topic"]["32"]["hit_rate@3"] == 0.0
    assert metrics["per_topic"]["32"]["relevant_count@3"] == 0
    assert metrics["per_topic"]["31"]["judged_count@3"] == 2
    assert metrics["per_topic"]["31"]["judged_rate@3"] == pytest.approx(2 / 3)
    assert metrics["per_topic"]["32"]["judged_count@3"] == 0
    assert metrics["per_topic"]["32"]["judged_rate@3"] == 0.0
    assert metrics["metrics"]["hit_rate@3"] == 0.5


def test_qrels_topic_without_ranked_rows_is_kept_in_macro_average(tmp_path):
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text("31 0 doc-a 4\n32 0 doc-b 4\n", encoding="utf-8")
    ranked = [RankedCandidate("31", "doc-a", 1, 1.0, "", [])]

    metrics = evaluate_ranked(
        ranked,
        parse_qrels(qrels_path),
        metric_names=["ndcg@10"],
        relevance_threshold=2,
        topic_ids=["31", "32"],
    )

    assert metrics["per_topic"]["31"]["ndcg@10"] == 1.0
    assert metrics["per_topic"]["32"]["ndcg@10"] == 0.0
    assert metrics["metrics"]["ndcg@10"] == 0.5


def test_pipeline_cache_dir_uses_shared_checkout_root_for_linked_worktree(tmp_path):
    shared = tmp_path / "shared"
    worktree = tmp_path / "worktree"
    git_dir = shared / ".git" / "worktrees" / "wt"
    git_dir.mkdir(parents=True)
    worktree.mkdir()
    (worktree / "AGENTS.md").write_text("# instructions\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")

    assert pipeline_cache_dir(worktree, "demo") == (
        shared / "cache" / "retrieval" / "pyserini_remote"
    )


def test_run_pipeline_writes_stage_outputs_with_fake_retriever(tmp_path):
    topics_path = tmp_path / "topics.data"
    topics_path.write_text("31\tExplain e-waste impacts.\n", encoding="utf-8")
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text("31 0 doc-a 4\n", encoding="utf-8")
    config_path = write_config(
        tmp_path / "config.yaml",
        f"""
experiment:
  id: rag25_fake_v1
submission:
  team_id: local-baseline
topics:
  path: {topics_path}
  format: tsv
query_understanding:
  variants:
    - name: original
      type: original_topic
retrievers:
  - name: fake
    type: fake
    query_variants: [original]
ranking:
  type: passthrough
evidence:
  type: top_k
  k: 2
  require_text: true
  allow_fewer: true
generation:
  type: placeholder
evaluation:
  kind: dev_projected_qrels
  qrels: {qrels_path}
  metrics: [ndcg@10, recall@100]
  relevance_threshold: 2
""",
    )

    class FakeRetriever:
        def retrieve(self, query):
            return [
                RetrievedCandidate(
                    query.topic_id,
                    query.variant_name,
                    "fake",
                    query.query_text,
                    "doc-a",
                    1,
                    9.0,
                    "Evidence text.",
                )
            ]

    cache_dirs = []

    def fake_factory(_config, cache_dir):
        cache_dirs.append(cache_dir)
        return FakeRetriever()

    result = run_pipeline(
        config_path,
        retriever_factories={"fake": fake_factory},
    )

    assert result.output_dir == tmp_path / "outputs" / "rag25_fake_v1"
    assert result.cache_dir == tmp_path / "cache" / "retrieval" / "pyserini_remote"
    assert cache_dirs == [result.cache_dir]
    assert (result.output_dir / "r_output_trec_rag_2026.tsv").read_text(
        encoding="utf-8"
    ).splitlines() == ["31 Q0 doc-a 1 9.0 rag25_fake_v1"]
    assert read_jsonl(result.output_dir / "stage_queries.jsonl")[0]["query_text"] == (
        "Explain e-waste impacts."
    )
    assert read_jsonl(result.output_dir / "rag_output_trec_rag_2026.jsonl")[0][
        "references"
    ] == ["doc-a"]
    assert json.loads((result.output_dir / "retrieval_metrics.json").read_text())["metrics"][
        "recall@100"
    ] == 1.0
    assert json.loads((result.output_dir / "run_metadata.json").read_text())["cache"][
        "retrieval"
    ][0]["status"] == "not_reported"


def test_run_pipeline_applies_cached_coverage_aware_reranker(tmp_path):
    topics_path = tmp_path / "topics.tsv"
    topics_path.write_text("31\tExplain banks.\n", encoding="utf-8")
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text("31 0 doc-b 4\n31 0 doc-a 0\n", encoding="utf-8")
    doc_scores = tmp_path / "doc_scores.jsonl"
    doc_scores.write_text(
        "\n".join(
            [
                json.dumps({"topic_id": "31", "docid": "doc-a", "score": 0.0}),
                json.dumps({"topic_id": "31", "docid": "doc-b", "score": 10.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    chunk_scores = tmp_path / "chunk_scores.jsonl"
    chunk_scores.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "topic_id": "31",
                        "docid": "doc-a",
                        "chunk_index": 0,
                        "start_char": 0,
                        "end_char": 1000,
                        "score": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "topic_id": "31",
                        "docid": "doc-b",
                        "chunk_index": 0,
                        "start_char": 0,
                        "end_char": 1000,
                        "score": 10.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = write_config(
        tmp_path / "config.yaml",
        f"""
experiment:
  id: rag25_fake_rerank_v1
submission:
  team_id: local-baseline
topics:
  path: {topics_path}
  format: tsv
query_understanding:
  variants:
    - name: original
      type: original_topic
retrievers:
  - name: fake
    type: fake
    query_variants: [original]
    hits: 50
ranking:
  type: coverage_aware_long_doc_aggregate
  reranker:
    model: mixedbread-ai/mxbai-rerank-base-v2
    score_source: cached_artifacts
    candidate_depth: 2
    document_score_path: {doc_scores}
    window_score_path: {chunk_scores}
    formula:
      long_document_weight: 0.5
      strongest_passage_weight: 0.5
      coverage_bonus_weight: 0.25
      relative_span_delta: 1.0
      support_cap: 6
      min_new_chars: 800
      top_window_weights: [0.55, 0.25, 0.13, 0.07]
evidence:
  type: top_k
  k: 1
  require_text: true
  allow_fewer: true
generation:
  type: placeholder
evaluation:
  kind: dev_projected_qrels
  qrels: {qrels_path}
  metrics: [ndcg@10, precision@10]
  relevance_threshold: 2
""",
    )

    class FakeRetriever:
        def retrieve(self, query):
            return [
                RetrievedCandidate(
                    query.topic_id,
                    query.variant_name,
                    "fake",
                    query.query_text,
                    "doc-a",
                    1,
                    9.0,
                    "A",
                ),
                RetrievedCandidate(
                    query.topic_id,
                    query.variant_name,
                    "fake",
                    query.query_text,
                    "doc-b",
                    2,
                    8.0,
                    "B",
                ),
                RetrievedCandidate(
                    query.topic_id,
                    query.variant_name,
                    "fake",
                    query.query_text,
                    "doc-c",
                    3,
                    7.0,
                    "C",
                ),
            ]

    result = run_pipeline(config_path, retriever_factories={"fake": lambda _config, _cache: FakeRetriever()})

    assert [(row.docid, row.rank) for row in result.ranked] == [
        ("doc-b", 1),
        ("doc-a", 2),
        ("doc-c", 3),
    ]
    assert result.ranked[0].provenance[-1]["ranker"] == "coverage_aware_long_doc_aggregate"
    assert result.ranked[-1].provenance[-1]["ranker"] == "bm25_tail_after_rerank"
    assert [row.score for row in result.ranked] == sorted(
        (row.score for row in result.ranked), reverse=True
    )
    assert read_jsonl(result.output_dir / "stage_evidence.jsonl")[0]["docid"] == "doc-b"
    assert result.metrics["metrics"]["ndcg@10"] == 1.0
    reranker_cache = result.run_metadata["cache"]["reranker"]
    assert reranker_cache["document_scores"]["sha256"] == hashlib.sha256(
        doc_scores.read_bytes()
    ).hexdigest()
    assert reranker_cache["window_scores"]["sha256"] == hashlib.sha256(
        chunk_scores.read_bytes()
    ).hexdigest()


def test_a_short_throttle_is_waited_out_instead_of_latching(tmp_path):
    """Losing a whole run to a one-second back-off is a ruinous trade."""
    query = QueryVariant("15", "original", "query", "topic")
    config = RetrieverConfig(
        name="bm25", type="pyserini_remote", query_variants=("original",),
        hits=10, index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 10, ())
        def __init__(self): self.calls = 0
        def search_raw(self, query_text, *, raw_sink):
            self.calls += 1
            raw = json.dumps(
                {
                    "api": "v1",
                    "index": "climbmix-400b",
                    "query": {"text": query_text},
                    "candidates": [],
                },
                separators=(",", ":"),
            ).encode()
            raw_sink(raw)
            if self.calls == 1:
                raise RemotePyseriniThrottled(1)
            return RemoteSearchResponse(raw, json.loads(raw), hashlib.sha256(raw).hexdigest())

    slept: list[float] = []
    retriever = PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=Client())
    retriever._sleep = slept.append

    retriever.retrieve(query)

    assert slept == [1], "the short back-off is honoured, once"
    assert not (tmp_path / "topic-state" / "15" / "ticket").exists(), "no latch"


def test_a_long_throttle_still_latches_with_a_ticket(tmp_path):
    query = QueryVariant("15", "original", "query", "topic")
    config = RetrieverConfig(
        name="bm25", type="pyserini_remote", query_variants=("original",),
        hits=10, index="climbmix-400b",
    )

    class Client:
        config = RemotePyseriniConfig("https://pyserini.test/search", None, 10, ())
        def search_raw(self, _query, *, raw_sink):
            raw_sink(b'{"api":"v1","index":"climbmix-400b","candidates":[]}')
            raise RemotePyseriniThrottled(600)

    retriever = PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=Client())
    retriever._sleep = lambda _seconds: pytest.fail("a long back-off must not be slept")

    with pytest.raises(RemotePyseriniThrottled):
        retriever.retrieve(query)

    assert (tmp_path / "topic-state" / "15" / "ticket").exists()
