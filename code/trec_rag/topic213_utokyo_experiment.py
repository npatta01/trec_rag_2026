"""Run a UTokyo-HitU-style retrieval and RAG experiment for Issue #19."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import yaml

from trec_rag.repo_env import find_repo_root, load_repo_env
from trec_rag.topic213_response_experiment import (
    OpenAICompatibleJsonClient,
    PassageCorpus,
    PassageUnit,
    _audit_section,
    _complete_validated,
    _normalize_sub_narrative,
    _require_nonempty_string,
    _sha256,
    _write_json,
    _write_jsonl,
    compute_evaluation_metrics,
    evaluate_nuggets,
    load_nuggets,
    load_passage_corpus,
    load_release_manifest,
    load_topic_narrative,
    render_evaluation_report,
    render_generated_response,
    validate_release_accounting,
)
from trec_rag.utokyo_retrieval import (
    RankedPassage,
    bm25_rank,
    corpus_sha256,
    dense_hyde_scores,
    load_score_cache,
    rank_scores,
    reciprocal_rank_fusion,
    score_cache_key,
    sliding_window_rerank,
    splade_scores,
    write_score_cache,
)


SCHEMA_VERSION = "topic213-utokyo-experiment-v1"
PROMPT_VERSION = "utokyo-issue19-local-adaptation-v1"
_LABEL_WORD_RE = re.compile(r"[a-z0-9]+")

HYDE_SYSTEM_PROMPT = """Generate a hypothetical answer of approximately 150 words for the supplied
query to improve dense retrieval accuracy. It does not need to be factually correct, but it must read
like a natural passage containing information relevant to the query. Return JSON with one string field
named `hypothetical_answer`. Do not use organizer nuggets or retrieved corpus evidence."""

KEYWORD_SYSTEM_PROMPT = """Generate approximately 10 related keywords for the supplied query to
improve BM25 retrieval accuracy. Include synonyms, related terms, technical terminology,
abbreviations, and common variations. Return JSON with one array field named `keywords`. Do not use
organizer nuggets or retrieved corpus evidence."""

RERANK_SYSTEM_PROMPT = """You are a passage reranker. Order all supplied candidates from most to
least useful for answering the query. Consider direct relevance, concrete factual detail, and source
specificity. Return JSON with one array field named `ranked_passage_ids`; it must be an exact
permutation of the supplied passage IDs. Use only the supplied candidate text."""

ANSWER_SYSTEM_PROMPT = """Write a cited research answer from the supplied top-ranked passages.
Return JSON with a `sections` array containing every allowed sub-narrative exactly once and in the
supplied order. Each section contains `sub_narrative` and a `claims` array. Each claim contains one
concise, atomic factual sentence in `text` and a nonempty `passage_ids` array. Cite only supplied
passage IDs that directly support the sentence. Use no outside knowledge and do not mention organizer
nuggets. A section may have an empty claims array when the selected evidence does not support it."""


def normalize_answer_label(value: object, allowed_labels: Sequence[str]) -> str:
    try:
        return _normalize_sub_narrative(value, allowed_labels)
    except ValueError:
        candidate = _require_nonempty_string(value, "sub_narrative")

        def normalized(label: str) -> str:
            cleaned = label.strip().strip('"').strip().casefold()
            if cleaned.startswith("new: "):
                cleaned = cleaned[5:]
            return " ".join(_LABEL_WORD_RE.findall(cleaned))

        candidate_normalized = normalized(candidate)
        scored = sorted(
            (
                difflib.SequenceMatcher(
                    None, candidate_normalized, normalized(label)
                ).ratio(),
                label,
            )
            for label in allowed_labels
        )
        best_score, best_label = scored[-1]
        second_score = scored[-2][0] if len(scored) > 1 else 0.0
        if best_score >= 0.85 and best_score - second_score >= 0.05:
            return best_label
        raise ValueError(f"unknown sub_narrative: {candidate}")


def prepare_query(
    *,
    client,
    query: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    temperature = float(config.get("temperature", 0.0))
    validation_attempts = int(config.get("validation_attempts", 3))

    def validate_hyde(value: Mapping[str, object]) -> str:
        return _require_nonempty_string(
            value.get("hypothetical_answer"), "hypothetical_answer"
        )

    hypothetical_answer = _complete_validated(
        client,
        stage="utokyo_hyde",
        system_prompt=HYDE_SYSTEM_PROMPT,
        payload={"query": query, "target_words": 150},
        max_tokens=int(config.get("hyde_max_tokens", 300)),
        temperature=temperature,
        validator=validate_hyde,
        validation_attempts=validation_attempts,
    )

    def validate_keywords(value: Mapping[str, object]) -> list[str]:
        raw = value.get("keywords")
        if not isinstance(raw, list):
            raise ValueError("keywords must be an array")
        keywords = list(
            dict.fromkeys(
                _require_nonempty_string(item, "keyword")
                for item in raw
            )
        )
        if not keywords:
            raise ValueError("keywords must not be empty")
        maximum = int(config.get("maximum_keywords", 15))
        if len(keywords) > maximum:
            raise ValueError(f"keywords must contain at most {maximum} items")
        return keywords

    keywords = _complete_validated(
        client,
        stage="utokyo_keywords",
        system_prompt=KEYWORD_SYSTEM_PROMPT,
        payload={"query": query, "target_keywords": 10},
        max_tokens=int(config.get("keyword_max_tokens", 200)),
        temperature=temperature,
        validator=validate_keywords,
        validation_attempts=validation_attempts,
    )
    return {
        "original_query": query,
        "hypothetical_answer": hypothetical_answer,
        "keywords": keywords,
        "bm25_expanded_query": " ".join([query, *keywords]),
    }


def _ranked_rows(rows: Sequence[RankedPassage]) -> list[dict[str, object]]:
    return [
        {"passage_id": row.passage_id, "rank": row.rank, "score": row.score}
        for row in rows
    ]


def _cached_model_scores(
    *,
    cache_dir: Path,
    cache_name: str,
    passages: Sequence[PassageUnit],
    corpus_hash: str,
    query: str,
    method: Mapping[str, object],
    compute: Callable[[], Sequence[float]],
) -> tuple[list[float], dict[str, object]]:
    passage_ids = [passage.passage_id for passage in passages]
    key = score_cache_key(corpus_hash=corpus_hash, query=query, method=method)
    path = cache_dir / f"{cache_name}__{key}.json"
    scores = load_score_cache(
        path, expected_passage_ids=passage_ids, expected_cache_key=key
    )
    cache_hit = scores is not None
    if scores is None:
        scores = [float(score) for score in compute()]
        if len(scores) != len(passages):
            raise ValueError(f"{cache_name} returned the wrong number of scores")
        write_score_cache(
            path,
            cache_key=key,
            passage_ids=passage_ids,
            scores=scores,
            metadata=method,
        )
    return scores, {
        "cache_hit": cache_hit,
        "cache_path": str(path),
        "cache_key": key,
    }


def retrieve_four_streams(
    *,
    corpus: PassageCorpus,
    query_preparation: Mapping[str, object],
    config: Mapping[str, object],
    cache_dir: Path,
) -> tuple[dict[str, list[RankedPassage]], dict[str, object]]:
    passages = corpus.passages
    ids = [passage.passage_id for passage in passages]
    stream_depth = min(int(config.get("stream_depth", 1000)), len(passages))
    alpha = float(config.get("hyde_alpha", 0.7))
    corpus_hash = corpus_sha256(passages)
    query = str(query_preparation["original_query"])
    hypothetical_answer = str(query_preparation["hypothetical_answer"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    streams: dict[str, list[RankedPassage]] = {
        "bm25_keywords": bm25_rank(
            passages,
            query=str(query_preparation["bm25_expanded_query"]),
            top_k=stream_depth,
        )
    }
    cache_records: dict[str, object] = {
        "bm25_keywords": {"cache_hit": False, "implementation": "rank_bm25.BM25Okapi"}
    }

    splade_config = config.get("splade")
    dense_configs = config.get("dense")
    if not isinstance(splade_config, Mapping) or not isinstance(dense_configs, list):
        raise ValueError("retrieval config requires splade mapping and dense array")
    splade_method = {
        "kind": "splade",
        **dict(splade_config),
    }
    splade_query = query
    splade_raw, cache_records["splade"] = _cached_model_scores(
        cache_dir=cache_dir,
        cache_name="splade",
        passages=passages,
        corpus_hash=corpus_hash,
        query=splade_query,
        method=splade_method,
        compute=lambda: splade_scores(
            passages,
            query=splade_query,
            model_name=str(splade_config["model"]),
            model_revision=(
                str(splade_config["revision"])
                if splade_config.get("revision")
                else None
            ),
            batch_size=int(splade_config.get("batch_size", 4)),
            device=str(splade_config.get("device", "cpu")),
            max_length=int(splade_config.get("max_length", 512)),
        ),
    )
    streams["splade"] = rank_scores(ids, splade_raw, top_k=stream_depth)

    for dense_config in dense_configs:
        if not isinstance(dense_config, Mapping):
            raise ValueError("dense retrieval entries must be mappings")
        name = _require_nonempty_string(dense_config.get("name"), "dense.name")
        method = {
            "kind": "dense_hyde_vector_mix",
            "alpha": alpha,
            **dict(dense_config),
        }
        cache_query = json.dumps(
            {"query": query, "hypothetical_answer": hypothetical_answer},
            ensure_ascii=False,
            sort_keys=True,
        )
        raw, cache_records[name] = _cached_model_scores(
            cache_dir=cache_dir,
            cache_name=name,
            passages=passages,
            corpus_hash=corpus_hash,
            query=cache_query,
            method=method,
            compute=lambda dense_config=dense_config: dense_hyde_scores(
                passages,
                query=query,
                hypothetical_answer=hypothetical_answer,
                model_name=str(dense_config["model"]),
                model_revision=(
                    str(dense_config["revision"])
                    if dense_config.get("revision")
                    else None
                ),
                alpha=alpha,
                batch_size=int(dense_config.get("batch_size", 16)),
                device=str(dense_config.get("device", "cpu")),
                max_seq_length=int(dense_config.get("max_length", 512)),
                query_prompt_name=(
                    str(dense_config["query_prompt_name"])
                    if dense_config.get("query_prompt_name")
                    else None
                ),
                query_prefix=str(dense_config.get("query_prefix", "")),
            ),
        )
        streams[name] = rank_scores(ids, raw, top_k=stream_depth)

    if set(streams) != {"bm25_keywords", "splade", "bge_hyde", "qwen_hyde"}:
        raise ValueError("UTokyo four-method run requires BM25, SPLADE, BGE, and Qwen streams")
    return streams, {
        "corpus_sha256": corpus_hash,
        "passages_scored_per_stream": len(passages),
        "stream_depth": stream_depth,
        "score_caches": cache_records,
    }


def rerank_fused_candidates(
    *,
    client,
    query: str,
    candidates: Sequence[RankedPassage],
    passage_lookup: Mapping[str, PassageUnit],
    config: Mapping[str, object],
) -> tuple[list[str], list[dict[str, object]]]:
    candidate_depth = min(int(config.get("candidate_depth", 200)), len(candidates))
    initial = [row.passage_id for row in candidates[:candidate_depth]]
    validation_attempts = int(config.get("validation_attempts", 3))
    temperature = float(config.get("temperature", 0.0))
    max_missing_ids = int(config.get("max_missing_ids", 2))
    normalizations: dict[tuple[int, int], dict[str, object]] = {}

    def rerank_window(
        passage_ids: Sequence[str], pass_index: int, window_index: int
    ) -> Sequence[str]:
        expected = set(passage_ids)
        payload = {
            "query": query,
            "current_order_best_to_worst": list(passage_ids),
            "candidates": [
                {
                    "passage_id": passage_id,
                    "document_id": passage_lookup[passage_id].document_id,
                    "text": passage_lookup[passage_id].text,
                }
                for passage_id in passage_ids
            ],
        }

        def validate(value: Mapping[str, object]) -> list[str]:
            ranked = value.get("ranked_passage_ids")
            if not isinstance(ranked, list):
                raise ValueError("ranked_passage_ids must be an array")
            result: list[str] = []
            corrected_ids: dict[str, str] = {}
            for raw_id in map(str, ranked):
                candidate = raw_id
                match = re.fullmatch(r"P(\d+)", raw_id)
                if candidate not in expected and match:
                    canonical = f"P{int(match.group(1)):06d}"
                    if canonical in expected:
                        candidate = canonical
                        corrected_ids[raw_id] = canonical
                result.append(candidate)
            if len(result) != len(set(result)) or any(item not in expected for item in result):
                raise ValueError("ranked_passage_ids contains a duplicate or unknown candidate")
            missing = [item for item in passage_ids if item not in result]
            if len(missing) > max_missing_ids:
                raise ValueError(
                    f"ranked_passage_ids omitted {len(missing)} candidates; maximum is {max_missing_ids}"
                )
            if missing or corrected_ids:
                result.extend(missing)
                normalizations[(pass_index, window_index)] = {
                    "policy": "canonicalize_numeric_ids_then_append_omissions",
                    "corrected_passage_ids": corrected_ids,
                    "omitted_passage_ids": missing,
                }
            return result

        return _complete_validated(
            client,
            stage=f"utokyo_rerank_p{pass_index:02d}_w{window_index:03d}",
            system_prompt=RERANK_SYSTEM_PROMPT,
            payload=payload,
            max_tokens=int(config.get("max_tokens", 250)),
            temperature=temperature,
            validator=validate,
            validation_attempts=validation_attempts,
        )

    order, audit = sliding_window_rerank(
        initial,
        rerank_window=rerank_window,
        window_size=int(config.get("window_size", 10)),
        stride=int(config.get("stride", 5)),
        num_passes=int(config.get("num_passes", 3)),
    )
    for row in audit:
        key = (int(row["pass"]), int(row["window"]))
        if key in normalizations:
            row["normalization"] = normalizations[key]
    return order, audit


def generate_top20_response(
    *,
    client,
    topic_id: str,
    narrative: str,
    corpus: PassageCorpus,
    selected_passage_ids: Sequence[str],
    release_accounting: Mapping[str, object],
    retrieval_record: Mapping[str, object],
    generation_config: Mapping[str, object],
    experiment_id: str,
    run_id: str,
    source_metadata: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    passage_lookup = {passage.passage_id: passage for passage in corpus.passages}
    if not selected_passage_ids or any(item not in passage_lookup for item in selected_passage_ids):
        raise ValueError("selected passage IDs must be a nonempty corpus subset")
    allowed_labels = list(corpus.sub_narratives)
    selected = [passage_lookup[item] for item in selected_passage_ids]
    max_claims = int(generation_config.get("max_answer_claims_per_section", 5))
    payload = {
        "task": "Answer the topic using only the final UTokyo-ranked passages.",
        "topic_narrative": narrative,
        "allowed_sub_narratives": allowed_labels,
        "maximum_claims_per_section": max_claims,
        "passages": [
            {
                "passage_id": passage.passage_id,
                "document_id": passage.document_id,
                "text": passage.text,
            }
            for passage in selected
        ],
    }

    def validate(value: Mapping[str, object]) -> list[dict[str, object]]:
        raw_sections = value.get("sections")
        if not isinstance(raw_sections, list) or any(
            not isinstance(section, Mapping) for section in raw_sections
        ):
            raise ValueError("sections must be an array of objects")
        by_label: dict[str, Mapping[str, object]] = {}
        for section in raw_sections:
            label = normalize_answer_label(section.get("sub_narrative"), allowed_labels)
            if label in by_label:
                raise ValueError(f"duplicate answer section: {label}")
            by_label[label] = section
        if set(by_label) != set(allowed_labels):
            raise ValueError("answer must contain every allowed sub-narrative exactly once")
        valid_ids = set(selected_passage_ids)
        normalized: list[dict[str, object]] = []
        for label in allowed_labels:
            raw_claims = by_label[label].get("claims")
            if not isinstance(raw_claims, list) or any(
                not isinstance(claim, Mapping) for claim in raw_claims
            ):
                raise ValueError("section claims must be an array of objects")
            if len(raw_claims) > max_claims:
                raise ValueError(f"section exceeds {max_claims} claims")
            claims: list[dict[str, object]] = []
            for claim in raw_claims:
                text = _require_nonempty_string(claim.get("text"), "answer claim text")
                passage_ids = claim.get("passage_ids")
                if not isinstance(passage_ids, list) or not passage_ids:
                    raise ValueError("answer passage_ids must be nonempty")
                ids = list(dict.fromkeys(map(str, passage_ids)))
                if any(item not in valid_ids for item in ids):
                    raise ValueError("answer cites a passage outside the selected top 20")
                claims.append({"text": text, "passage_ids": ids})
            normalized.append({"sub_narrative": label, "claims": claims})
        return normalized

    raw_sections = _complete_validated(
        client,
        stage="utokyo_generate_top20",
        system_prompt=ANSWER_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=int(generation_config.get("answer_max_tokens", 1800)),
        temperature=float(generation_config.get("temperature", 0.0)),
        validator=validate,
        validation_attempts=int(generation_config.get("validation_attempts", 3)),
    )

    evidence_ids: dict[tuple[str, str], str] = {}
    evidence_ledger: list[dict[str, object]] = []
    sections: list[dict[str, object]] = []
    answer_index = 0
    for raw_section in raw_sections:
        label = str(raw_section["sub_narrative"])
        answer_claims: list[dict[str, object]] = []
        for raw_claim in raw_section["claims"]:
            answer_index += 1
            linked_evidence_ids: list[str] = []
            supporting_passages: list[dict[str, object]] = []
            for passage_id in raw_claim["passage_ids"]:
                passage = passage_lookup[str(passage_id)]
                key = (label, passage.passage_id)
                if key not in evidence_ids:
                    evidence_id = f"UTE{len(evidence_ledger) + 1:03d}"
                    evidence_ids[key] = evidence_id
                    evidence_ledger.append(
                        {
                            "claim_id": evidence_id,
                            "text": passage.text,
                            "sub_narrative": label,
                            "passage_ids": [passage.passage_id],
                            "document_ids": [passage.document_id],
                            "supporting_passages": [
                                {
                                    "passage_id": passage.passage_id,
                                    "document_id": passage.document_id,
                                    "text": passage.text,
                                }
                            ],
                        }
                    )
                linked_evidence_ids.append(evidence_ids[key])
                supporting_passages.append(
                    {
                        "passage_id": passage.passage_id,
                        "document_id": passage.document_id,
                        "text": passage.text,
                    }
                )
            answer_claims.append(
                {
                    "claim_id": f"A{answer_index:03d}",
                    "text": raw_claim["text"],
                    "evidence_claim_ids": linked_evidence_ids,
                    "passage_ids": list(raw_claim["passage_ids"]),
                    "document_ids": list(
                        dict.fromkeys(row["document_id"] for row in supporting_passages)
                    ),
                    "supporting_passages": supporting_passages,
                }
            )
        sections.append({"sub_narrative": label, "claims": answer_claims})

    support_audit: list[dict[str, object]] = []
    for section in sections:
        support_audit.extend(
            _audit_section(
                client,
                sub_narrative=str(section["sub_narrative"]),
                claims=section["claims"],
                max_tokens=int(generation_config.get("audit_max_tokens", 350)),
                temperature=float(generation_config.get("audit_temperature", 0.0)),
                validation_attempts=int(generation_config.get("validation_attempts", 3)),
            )
        )

    generation = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "topic_id": topic_id,
        "narrative": narrative,
        "model": client.model,
        "temperature": float(generation_config.get("temperature", 0.0)),
        "context_policy": {
            "retrieval_corpus": "all released passages",
            "generation_evidence": "final reranked top 20 passages",
            "nuggets_available_during_generation": False,
        },
        "input_accounting": dict(release_accounting),
        "source_metadata": dict(source_metadata),
        "generation_config": dict(generation_config),
        "retrieval": dict(retrieval_record),
        "selected_passages": [
            {
                "rank": rank,
                "passage_id": passage.passage_id,
                "document_id": passage.document_id,
                "released_sub_narrative": passage.sub_narrative,
                "text": passage.text,
            }
            for rank, passage in enumerate(selected, 1)
        ],
        "prompts": {
            "hyde_system": HYDE_SYSTEM_PROMPT,
            "keyword_system": KEYWORD_SYSTEM_PROMPT,
            "rerank_system": RERANK_SYSTEM_PROMPT,
            "answer_system": ANSWER_SYSTEM_PROMPT,
        },
        "evidence_ledger": evidence_ledger,
        "sections": sections,
    }
    return generation, support_audit


def _resolve(repo_root: Path, value: object, label: str) -> Path:
    path = Path(_require_nonempty_string(value, label))
    return path if path.is_absolute() else repo_root / path


def _source_metadata(
    *, repo_root: Path, inputs: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, path in paths.items():
        result[f"{name}_path"] = str(path.relative_to(repo_root))
        result[f"{name}_sha256"] = _sha256(path)
    result["passages_release_url"] = inputs.get("passages_release_url")
    result["trec_rag_data_revision"] = inputs.get("trec_rag_data_revision")
    result["utokyo_paper"] = inputs.get("utokyo_paper")
    return result


def _render_utokyo_report(
    *, generation: Mapping[str, object], metrics: Mapping[str, object]
) -> str:
    base = render_evaluation_report(generation=generation, metrics=metrics)
    base = base.replace(
        "# Topic 213 response-generation evaluation",
        "# Topic 213 UTokyo-style RAG evaluation",
        1,
    )
    base = base.replace(
        "The full released passage packet was consumed during generation. Two eligible documents are absent because their original provider calls failed before inference, and this experiment does not fabricate or replace their evidence.",
        "All released passages were scored by every retrieval stream. Generation used only the final reranked top 20. Two eligible documents are absent because their original provider calls failed before inference.",
    )
    retrieval = metrics["retrieval"]
    assert isinstance(retrieval, Mapping)
    lines = [
        base.rstrip(),
        "",
        "## UTokyo retrieval adaptation",
        "",
        f"- Passages scored in every stream: {retrieval['passages_scored_per_stream']}",
        f"- RRF candidates retained: {retrieval['rrf_candidates']}",
        f"- Sliding-window candidates: {retrieval['reranked_candidates']}",
        f"- Final generation passages: {retrieval['generation_passages']}",
        f"- Final generation documents: {retrieval['generation_documents']}",
        f"- Reranking windows: {retrieval['reranking_windows']}",
        "",
        "The paper's title and URL reranking fields are unavailable in the Issue #19 passage release, so this run uses document ID and verbatim passage text. Local Qwen replaces GPT-4.1 and GPT-4.1-mini; the retrieval models and the paper's alpha=0.7, RRF k=60, top-1000 streams, top-200 reranking, 10/5/3 windows, and top-20 generation cutoff are retained.",
        "",
    ]
    return "\n".join(lines)


def run_from_config(config_path: Path) -> Path:
    repo_root = find_repo_root(config_path.parent)
    load_repo_env(repo_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("experiment config must be a YAML mapping")
    required = [
        "experiment",
        "inputs",
        "query_preprocessing",
        "retrieval",
        "reranking",
        "generation",
        "evaluation",
    ]
    if any(not isinstance(config.get(key), Mapping) for key in required):
        raise ValueError(f"config requires mapping sections: {', '.join(required)}")
    experiment = config["experiment"]
    inputs = config["inputs"]
    query_config = config["query_preprocessing"]
    retrieval_config = config["retrieval"]
    reranking_config = config["reranking"]
    generation_config = config["generation"]
    evaluation_config = config["evaluation"]
    assert all(
        isinstance(item, Mapping)
        for item in (
            experiment,
            inputs,
            query_config,
            retrieval_config,
            reranking_config,
            generation_config,
            evaluation_config,
        )
    )

    experiment_id = _require_nonempty_string(experiment.get("id"), "experiment.id")
    run_id = _require_nonempty_string(experiment.get("run_id"), "experiment.run_id")
    topic_id = str(experiment.get("topic_id", "213"))
    output_dir = _resolve(repo_root, experiment.get("output_dir"), "experiment.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "passages": _resolve(repo_root, inputs.get("passages"), "inputs.passages"),
        "release_manifest": _resolve(
            repo_root, inputs.get("release_manifest"), "inputs.release_manifest"
        ),
        "topics": _resolve(repo_root, inputs.get("topics"), "inputs.topics"),
        "nuggets": _resolve(repo_root, inputs.get("nuggets"), "inputs.nuggets"),
        "utokyo_paper": _resolve(
            repo_root, inputs.get("utokyo_paper"), "inputs.utokyo_paper"
        ),
    }
    for key, path in paths.items():
        expected = inputs.get(f"{key}_sha256")
        if expected and str(expected) != _sha256(path):
            raise ValueError(f"{key}_sha256 does not match {path}")

    corpus = load_passage_corpus(paths["passages"])
    release_manifest = load_release_manifest(paths["release_manifest"])
    accounting = validate_release_accounting(corpus, release_manifest)
    narrative = load_topic_narrative(paths["topics"], topic_id=topic_id)

    api_base = os.environ.get(
        str(generation_config.get("api_base_env", "LITELLM_BASE_URL")),
        str(generation_config.get("api_base", "http://localhost:4000/v1")),
    )
    model = os.environ.get(
        str(generation_config.get("model_env", "LITELLM_MODEL")),
        str(generation_config.get("model", "qwen-local")),
    )
    api_key = os.environ.get(
        str(generation_config.get("api_key_env", "LITELLM_API_KEY")),
        str(generation_config.get("api_key", "none")),
    )
    client = OpenAICompatibleJsonClient(
        api_base=api_base,
        model=model,
        api_key=api_key,
        checkpoint_dir=output_dir / "checkpoints",
        call_log_path=output_dir / "llm_calls.jsonl",
        timeout_seconds=float(generation_config.get("timeout_seconds", 240)),
        max_attempts=int(generation_config.get("http_max_attempts", 4)),
    )

    query_preparation = prepare_query(
        client=client, query=narrative, config=query_config
    )
    cache_dir = _resolve(repo_root, retrieval_config.get("cache_dir"), "retrieval.cache_dir")
    streams, retrieval_runtime = retrieve_four_streams(
        corpus=corpus,
        query_preparation=query_preparation,
        config=retrieval_config,
        cache_dir=cache_dir,
    )
    rrf_k = int(retrieval_config.get("rrf_k", 60))
    fused_depth = int(retrieval_config.get("fused_depth", 1000))
    fused, provenance = reciprocal_rank_fusion(streams, k=rrf_k, top_k=fused_depth)
    two_method, _ = reciprocal_rank_fusion(
        {"splade": streams["splade"], "qwen_hyde": streams["qwen_hyde"]},
        k=rrf_k,
        top_k=fused_depth,
    )
    passage_lookup = {passage.passage_id: passage for passage in corpus.passages}
    reranked_ids, rerank_audit = rerank_fused_candidates(
        client=client,
        query=narrative,
        candidates=fused,
        passage_lookup=passage_lookup,
        config=reranking_config,
    )
    generation_depth = min(int(generation_config.get("evidence_depth", 20)), len(reranked_ids))
    selected_ids = reranked_ids[:generation_depth]
    retrieval_record = {
        "query_preparation": query_preparation,
        "configuration": {
            "hyde_alpha": float(retrieval_config.get("hyde_alpha", 0.7)),
            "rrf_k": rrf_k,
            "stream_depth": int(retrieval_config.get("stream_depth", 1000)),
            "fused_depth": fused_depth,
            "reranking": dict(reranking_config),
            "generation_depth": generation_depth,
        },
        "runtime": retrieval_runtime,
        "streams": {name: _ranked_rows(rows) for name, rows in streams.items()},
        "two_method_rrf": _ranked_rows(two_method),
        "four_method_rrf": [
            {**row, "provenance": provenance[str(row["passage_id"])]}
            for row in _ranked_rows(fused)
        ],
        "reranked_passage_ids": reranked_ids,
        "rerank_audit": rerank_audit,
        "selected_passage_ids": selected_ids,
        "implementation_adaptations": [
            "The Issue #19 release has no title or URL fields; reranking uses document ID and passage text.",
            "Local Qwen replaces GPT-4.1 for query expansion and GPT-4.1-mini for reranking.",
            "rank-bm25 replaces Anserini because the released passage packet is a local experiment corpus.",
        ],
    }
    _write_json(output_dir / "retrieval_run.json", retrieval_record)

    generation, support_audit = generate_top20_response(
        client=client,
        topic_id=topic_id,
        narrative=narrative,
        corpus=corpus,
        selected_passage_ids=selected_ids,
        release_accounting=accounting,
        retrieval_record={
            "retrieval_run_path": "retrieval_run.json",
            "retrieval_run_sha256": _sha256(output_dir / "retrieval_run.json"),
            "selected_passage_ids": selected_ids,
        },
        generation_config=generation_config,
        experiment_id=experiment_id,
        run_id=run_id,
        source_metadata=_source_metadata(
            repo_root=repo_root, inputs=inputs, paths=paths
        ),
    )

    generation_path = output_dir / "response_generation.json"
    response_path = output_dir / "generated_response.md"
    support_path = output_dir / "claim_support_audit.jsonl"
    _write_json(generation_path, generation)
    response_text = render_generated_response(generation)
    response_path.write_text(response_text, encoding="utf-8")
    _write_jsonl(support_path, support_audit)
    frozen_generation_sha256 = _sha256(generation_path)
    frozen_response_sha256 = _sha256(response_path)

    # The organizer nuggets are deliberately unread until generation is frozen.
    nuggets = load_nuggets(
        paths["nuggets"],
        topic_id=topic_id,
        expected_count=int(evaluation_config.get("expected_nugget_count", 50)),
    )
    comparison = evaluate_nuggets(
        nuggets=nuggets,
        generation=generation,
        client=client,
        evaluation_config=evaluation_config,
    )
    _write_jsonl(output_dir / "nugget_comparison.jsonl", comparison)
    metrics = compute_evaluation_metrics(
        comparison, support_audit, response_text=response_text
    )
    selected_documents = {
        passage_lookup[passage_id].document_id for passage_id in selected_ids
    }
    metrics.update(
        {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "topic_id": topic_id,
            "frozen_generation_sha256": frozen_generation_sha256,
            "frozen_response_sha256": frozen_response_sha256,
            "retrieval": {
                "passages_scored_per_stream": len(corpus.passages),
                "rrf_candidates": len(fused),
                "reranked_candidates": len(reranked_ids),
                "reranking_windows": len(rerank_audit),
                "generation_passages": len(selected_ids),
                "generation_documents": len(selected_documents),
                "selected_released_sub_narratives": dict(
                    Counter(passage_lookup[item].sub_narrative for item in selected_ids)
                ),
            },
        }
    )
    _write_json(output_dir / "metrics.json", metrics)
    (output_dir / "evaluation_report.md").write_text(
        _render_utokyo_report(generation=generation, metrics=metrics),
        encoding="utf-8",
    )
    _write_json(output_dir / "config.resolved.json", json.loads(json.dumps(config)))
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    output_dir = run_from_config(args.config.resolve())
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
