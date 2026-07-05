import json
import urllib.error
from io import BytesIO

import pytest

from trec_rag.evaluation import evaluate_ranked, parse_qrels
from trec_rag.evidence import select_top_k_evidence
from trec_rag.generation import generate_placeholder_rag
from trec_rag.pipeline import pipeline_cache_dir, run_pipeline
from trec_rag.litellm_facets import LiteLLMFacetGenerator, UrllibJsonTransport
from trec_rag.pipeline_config import RetrieverConfig, load_pipeline_config
from trec_rag.pipeline_models import (
    QueryVariant,
    RankedCandidate,
    RetrievedCandidate,
)
from trec_rag.query_understanding import build_query_variants
from trec_rag.ranking import passthrough_rank, rrf_rank
from trec_rag.remote_pyserini import RemotePyseriniConfig
from trec_rag.retrievers import (
    PyseriniRemoteRetriever,
    cache_path,
    normalize_retrieved_candidates,
    request_cache_key,
)
from trec_rag.topics import Topic


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_config(path, body):
    path.write_text(body, encoding="utf-8")
    return path


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


def test_config_loads_title_llm_facets_and_rrf(tmp_path):
    config_path = write_config(
        tmp_path / "config.yaml",
        """
experiment: {id: rag25_facets_v1}
submission: {team_id: local-baseline}
topics: {path: topics.tsv, format: tsv}
query_understanding:
  variants:
    - {name: title, type: title}
    - {name: facets, type: llm_facets, provider: litellm, max_facets: 7, cache: false}
retrievers:
  - name: climbmix_bm25
    type: pyserini_remote
    query_variants: [title, facets]
    hits: 100
    index: climbmix-400b
    request_delay_seconds: 2.5
ranking:
  type: rrf
  rrf_k: 42
  stream_weights: {title: 1.0, facets: 0.5}
  dedupe: {by: docid, keep: best_rank, preserve_provenance: true}
evidence: {type: top_k, k: 5, allow_fewer: true}
generation: {type: placeholder}
""",
    )

    config = load_pipeline_config(config_path)

    assert [(row.name, row.type) for row in config.query_variants] == [
        ("title", "title"),
        ("facets", "llm_facets"),
    ]
    assert config.query_variants[1].provider == "litellm"
    assert config.query_variants[1].max_facets == 7
    assert config.query_variants[1].cache is False
    assert config.retrievers[0].request_delay_seconds == 2.5
    assert config.ranking.type == "rrf"
    assert config.ranking.rrf_k == 42
    assert config.ranking.stream_weights == {"title": 1.0, "facets": 0.5}


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


def test_title_and_litellm_facet_query_variants():
    topic = Topic(
        id="31",
        title="Electronic waste impacts",
        narrative="Explain health risks and recycling responses for e-waste.",
    )

    class FakeFacetGenerator:
        def generate(self, topic, *, variant_name, max_facets, cache):
            assert topic.id == "31"
            assert variant_name == "facets"
            assert max_facets == 2
            assert cache is True
            return [
                {"search_query": "e-waste health risks"},
                {"search_query": "e-waste recycling responses"},
            ]

    variants = build_query_variants(
        topic,
        variant_configs=[
            {"name": "title", "type": "title"},
            {"name": "facets", "type": "llm_facets", "max_facets": 2, "cache": True},
        ],
        facet_generator=FakeFacetGenerator(),
    )

    assert variants == [
        QueryVariant("31", "title", "Electronic waste impacts", "title"),
        QueryVariant("31", "facets", "e-waste health risks", "llm_facet"),
        QueryVariant("31", "facets", "e-waste recycling responses", "llm_facet"),
    ]


def test_litellm_facet_generator_rejects_invalid_json(tmp_path):
    class FakeTransport:
        def post_json(self, _url, _payload, _headers, _timeout):
            return {"choices": [{"message": {"content": "not-json"}}]}

    generator = LiteLLMFacetGenerator(
        base_url="http://litellm.test/v1",
        model="qwen-local",
        cache_dir=tmp_path,
        transport=FakeTransport(),
    )

    with pytest.raises(ValueError, match="valid JSON"):
        generator.generate(
            Topic("31", "E-waste", "Explain e-waste impacts."),
            variant_name="facets",
            max_facets=5,
            cache=False,
        )


def test_litellm_facet_generator_parses_and_caches_facets(tmp_path):
    class FakeTransport:
        def __init__(self):
            self.calls = 0

        def post_json(self, _url, _payload, _headers, _timeout):
            self.calls += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```json\n"
                                "{\"facets\":[{\"search_query\":\"e-waste health risks\"},"
                                "{\"query\":\"battery recycling policy\"}]}\n"
                                "```"
                            )
                        }
                    }
                ]
            }

    transport = FakeTransport()
    generator = LiteLLMFacetGenerator(
        base_url="http://litellm.test/v1",
        model="qwen-local",
        cache_dir=tmp_path,
        transport=transport,
    )

    topic = Topic("31", "E-waste", "Explain e-waste impacts.")
    first = generator.generate(topic, variant_name="facets", max_facets=2, cache=True)
    second = generator.generate(topic, variant_name="facets", max_facets=2, cache=True)

    assert first == [
        {"search_query": "e-waste health risks"},
        {"search_query": "battery recycling policy"},
    ]
    assert second == first
    assert transport.calls == 1


def test_litellm_transport_includes_http_error_body(monkeypatch):
    def fake_urlopen(_request, timeout):
        assert timeout == 3
        raise urllib.error.HTTPError(
            url="http://litellm.test/v1/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=BytesIO(b'{"error":"Cannot connect to host localhost:8000"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="localhost:8000"):
        UrllibJsonTransport().post_json(
            "http://litellm.test/v1/chat/completions",
            {"model": "qwen-local"},
            {"Content-Type": "application/json"},
            3,
        )


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


def test_pyserini_remote_retriever_uses_yaml_index_over_env_index(tmp_path, monkeypatch):
    monkeypatch.delenv("INDEX_URL", raising=False)
    monkeypatch.setenv("PYSERINI_INDEX", "other-index")
    monkeypatch.setenv("PYSERINI_BASE_URL", "https://pyserini.test")
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

    cache_file = tmp_path / cache_path(
        query.topic_id,
        query.variant_name,
        config.name,
        request_cache_key(config, query, index_url=FailingClient.config.index_url),
    )
    cache_file.write_text(
        json.dumps(
            {
                "response": {
                    "candidates": [
                        {"rank": 1, "docid": "doc-cached", "score": 7.0, "doc": {"contents": "Cached"}}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    retriever = PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=FailingClient())

    candidates = retriever.retrieve(query)

    assert [candidate.docid for candidate in candidates] == ["doc-cached"]


def test_pyserini_remote_retriever_cache_false_bypasses_cache_and_does_not_write(tmp_path):
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
                    {"rank": 1, "docid": "doc-fresh", "score": 9.0, "doc": {"contents": "Fresh"}}
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
                        {"rank": 1, "docid": "doc-stale", "score": 1.0, "doc": {"contents": "Stale"}}
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
    assert sorted(path.name for path in tmp_path.iterdir()) == [stale_cache.name]


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


def test_passthrough_rejects_multiple_retrieval_streams():
    candidates = [
        RetrievedCandidate("31", "original", "bm25", "query", "doc-a", 1, 9.0, "A"),
        RetrievedCandidate("31", "rewrite", "bm25", "query", "doc-b", 1, 8.0, "B"),
    ]

    with pytest.raises(ValueError, match="passthrough.*one retrieval stream"):
        passthrough_rank(candidates)


def test_rrf_rank_fuses_query_streams_and_preserves_provenance():
    candidates = [
        RetrievedCandidate("31", "title", "bm25", "e-waste", "doc-a", 1, 10.0, "A"),
        RetrievedCandidate("31", "title", "bm25", "e-waste", "doc-b", 2, 9.0, "B"),
        RetrievedCandidate("31", "facets", "bm25", "health risks", "doc-b", 1, 8.0, "B facet"),
        RetrievedCandidate("31", "facets", "bm25", "recycling", "doc-c", 1, 7.0, "C"),
    ]

    ranked = rrf_rank(candidates, k=60)

    assert [(row.docid, row.rank) for row in ranked] == [
        ("doc-b", 1),
        ("doc-a", 2),
        ("doc-c", 3),
    ]
    assert ranked[0].score == pytest.approx((1 / 62) + (1 / 61))
    assert {item["query_text"] for item in ranked[0].provenance} == {"e-waste", "health risks"}


def test_weighted_rrf_can_make_original_topic_stream_dominate_facets():
    candidates = [
        RetrievedCandidate("31", "original", "bm25", "full narrative", "doc-original", 1, 10.0, "A"),
        RetrievedCandidate("31", "facets", "bm25", "facet one", "doc-facet", 1, 9.0, "B"),
        RetrievedCandidate("31", "facets", "bm25", "facet two", "doc-facet", 1, 8.0, "B again"),
        RetrievedCandidate("31", "facets", "bm25", "facet three", "doc-facet", 1, 7.0, "B third"),
    ]

    ranked = rrf_rank(candidates, k=60, stream_weights={"original": 4.0, "facets": 0.25})

    assert [(row.docid, row.rank) for row in ranked] == [
        ("doc-original", 1),
        ("doc-facet", 2),
    ]
    assert ranked[0].score == pytest.approx(4.0 / 61)
    assert ranked[1].score == pytest.approx(3 * (0.25 / 61))


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


def test_pipeline_cache_dir_uses_shared_checkout_root_for_linked_worktree(tmp_path):
    shared = tmp_path / "shared"
    worktree = tmp_path / "worktree"
    git_dir = shared / ".git" / "worktrees" / "wt"
    git_dir.mkdir(parents=True)
    worktree.mkdir()
    (worktree / "AGENTS.md").write_text("# instructions\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")

    assert pipeline_cache_dir(worktree, "demo") == shared / "outputs" / "demo" / "cache"


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
    assert result.cache_dir == tmp_path / "outputs" / "rag25_fake_v1" / "cache"
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


def test_run_pipeline_writes_rrf_outputs_with_fake_facets_and_fake_retriever(tmp_path):
    topics_path = tmp_path / "topics.jsonl"
    topics_path.write_text(
        json.dumps(
            {
                "id": "31",
                "title": "Electronic waste impacts",
                "narrative": "Explain health risks and recycling responses for e-waste.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text("31 0 doc-overlap 4\n", encoding="utf-8")
    config_path = write_config(
        tmp_path / "config.yaml",
        f"""
experiment: {{id: rag25_fake_facets_v1}}
submission: {{team_id: local-baseline}}
topics: {{path: {topics_path}, format: jsonl}}
query_understanding:
  variants:
    - {{name: title, type: title}}
    - {{name: facets, type: llm_facets, max_facets: 2, cache: true}}
retrievers:
  - name: fake
    type: fake
    query_variants: [title, facets]
    hits: 10
ranking:
  type: rrf
  rrf_k: 60
evidence: {{type: top_k, k: 2, require_text: true, allow_fewer: true}}
generation: {{type: placeholder}}
evaluation:
  qrels: {qrels_path}
  metrics: [recall@10]
""",
    )

    class FakeFacetGenerator:
        def generate(self, _topic, *, variant_name, max_facets, cache):
            assert (variant_name, max_facets, cache) == ("facets", 2, True)
            return [
                {"search_query": "e-waste health risks"},
                {"search_query": "e-waste recycling responses"},
            ]

    class FakeRetriever:
        def retrieve(self, query):
            docs_by_query = {
                "Electronic waste impacts": [
                    ("doc-title", 1, 10.0, "Title-only evidence."),
                    ("doc-overlap", 2, 9.0, "Overlap evidence from title."),
                ],
                "e-waste health risks": [
                    ("doc-overlap", 1, 8.0, "Overlap evidence from facet."),
                ],
                "e-waste recycling responses": [
                    ("doc-facet", 1, 7.0, "Facet-only evidence."),
                ],
            }
            return [
                RetrievedCandidate(
                    query.topic_id,
                    query.variant_name,
                    "fake",
                    query.query_text,
                    docid,
                    rank,
                    score,
                    text,
                )
                for docid, rank, score, text in docs_by_query[query.query_text]
            ]

    result = run_pipeline(
        config_path,
        retriever_factories={"fake": lambda _config, _cache_dir: FakeRetriever()},
        facet_generator=FakeFacetGenerator(),
    )

    queries = read_jsonl(result.output_dir / "stage_queries.jsonl")
    assert [row["query_text"] for row in queries] == [
        "Electronic waste impacts",
        "e-waste health risks",
        "e-waste recycling responses",
    ]
    assert (result.output_dir / "r_output_trec_rag_2026.tsv").read_text(
        encoding="utf-8"
    ).splitlines()[0].startswith("31 Q0 doc-overlap 1 ")
    assert result.ranked[0].docid == "doc-overlap"
    assert len(result.ranked[0].provenance) == 2
