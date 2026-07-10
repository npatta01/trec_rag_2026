"""Build cached cross-encoder reranker scores from shared retrieval caches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
from trec_rag.pipeline import pipeline_cache_dir
from trec_rag.pipeline_config import PipelineConfig, RetrieverConfig, load_pipeline_config
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate, jsonable
from trec_rag.query_understanding import build_query_variants
from trec_rag.ranking import passthrough_rank
from trec_rag.remote_config import RemotePyseriniConfig
from trec_rag.repo_env import load_repo_env, repo_cache_root, shared_checkout_root
from trec_rag.retrievers import cache_path, normalize_retrieved_candidates, request_cache_key
from trec_rag.topics import Topic, load_topics


DEFAULT_INDEX_URL = "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    if topic_id.isdigit():
        return (0, int(topic_id))
    return (1, topic_id)


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScoreCacheContext:
    backend: str
    model: str
    max_length: int
    score_kind: str

    @property
    def path_parts(self) -> tuple[str, str, str, str]:
        return (
            _slug(self.backend),
            _slug(self.model),
            f"max_length_{self.max_length}",
            f"{_slug(self.score_kind)}.jsonl",
        )


class GlobalScoreCache:
    """Content-addressed cross-encoder score cache shared across experiments."""

    schema_version = 1

    def __init__(self, root_dir: Path, context: ScoreCacheContext) -> None:
        self.context = context
        self.path = root_dir.joinpath(*context.path_parts)
        self.scores = self._load()

    def _load(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not self.path.exists():
            return scores
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path}:{line_number}: invalid JSONL row") from exc
            if row.get("schema_version") != self.schema_version:
                continue
            scores[str(row["cache_key"])] = float(row["score"])
        return scores

    def cache_key(self, *, query_text: str, text: str) -> str:
        payload = {
            "schema_version": self.schema_version,
            "backend": self.context.backend,
            "model": self.context.model,
            "max_length": self.context.max_length,
            "score_kind": self.context.score_kind,
            "query_sha256": _sha256_text(query_text),
            "text_sha256": _sha256_text(text),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, *, query_text: str, text: str) -> float | None:
        return self.scores.get(self.cache_key(query_text=query_text, text=text))

    def add_many(self, rows: Iterable[tuple[str, str, float]]) -> int:
        materialized: list[dict[str, Any]] = []
        for query_text, text, score in rows:
            cache_key = self.cache_key(query_text=query_text, text=text)
            if cache_key in self.scores:
                continue
            query_hash = _sha256_text(query_text)
            text_hash = _sha256_text(text)
            self.scores[cache_key] = float(score)
            materialized.append(
                {
                    "schema_version": self.schema_version,
                    "backend": self.context.backend,
                    "model": self.context.model,
                    "max_length": self.context.max_length,
                    "score_kind": self.context.score_kind,
                    "cache_key": cache_key,
                    "query_sha256": query_hash,
                    "text_sha256": text_hash,
                    "score": float(score),
                }
            )
        return _append_jsonl(self.path, materialized)


def global_score_cache_dir(root_dir: Path) -> Path:
    return repo_cache_root(root_dir) / "reranker" / "score_cache"


def shared_output_path(root_dir: Path, path: Path) -> Path:
    """Resolve configured score output paths into the shared checkout when possible."""
    shared_root = shared_checkout_root(root_dir) or root_dir
    if not path.is_absolute():
        return shared_root / path
    try:
        return shared_root / path.relative_to(root_dir)
    except ValueError:
        return path


def _read_jsonl_keys(path: Path, fields: Sequence[str]) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    if not path.exists():
        return keys
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        keys.add(tuple(str(row[field]) for field in fields))
    return keys


def _read_document_scores(path: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    if not path.exists():
        return scores
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        scores[(str(row["topic_id"]), str(row["docid"]))] = float(row["score"])
    return scores


def _read_window_scores(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    scores: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return scores
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        key = (str(row["topic_id"]), str(row["docid"]), int(row["chunk_index"]))
        scores[key] = row
    return scores


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    if not materialized:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        for row in materialized:
            sink.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
    return len(materialized)


def _seed_document_score_cache(
    *,
    topic: Topic,
    candidates: list[RetrievedCandidate],
    existing_scores: dict[tuple[str, str], float],
    score_cache: GlobalScoreCache,
) -> int:
    return score_cache.add_many(
        (candidate.query_text, candidate.text, existing_scores[(topic.id, candidate.docid)])
        for candidate in candidates
        if (topic.id, candidate.docid) in existing_scores
        and score_cache.get(query_text=candidate.query_text, text=candidate.text) is None
    )


def _seed_window_score_cache(
    *,
    topic: Topic,
    candidates: list[RetrievedCandidate],
    existing_scores: dict[tuple[str, str, int], dict[str, Any]],
    score_cache: GlobalScoreCache,
    chunker: SemanticTextChunker,
) -> int:
    rows: list[tuple[str, str, float]] = []
    existing_docids = {key[:2] for key in existing_scores}
    for candidate in candidates:
        topic_doc_prefix = (topic.id, candidate.docid)
        if topic_doc_prefix not in existing_docids:
            continue
        chunks = chunker.split_text(candidate.text, document_id=candidate.docid)
        for chunk_index, chunk in enumerate(chunks):
            row = existing_scores.get((topic.id, candidate.docid, chunk_index))
            if row is None:
                continue
            if int(row["start_char"]) != chunk.start_char or int(row["end_char"]) != chunk.end_char:
                continue
            if score_cache.get(query_text=candidate.query_text, text=chunk.text) is None:
                rows.append((candidate.query_text, chunk.text, float(row["score"])))
    return score_cache.add_many(rows)


def _load_cached_candidates(
    *,
    query: QueryVariant,
    retriever: RetrieverConfig,
    cache_dir: Path,
    index_url: str,
) -> list[RetrievedCandidate]:
    request_key = request_cache_key(retriever, query, index_url=index_url)
    candidate_cache = cache_dir / cache_path(
        query.topic_id,
        query.variant_name,
        retriever.name,
        request_key,
    )
    if not candidate_cache.exists():
        raise FileNotFoundError(
            f"missing retrieval cache for topic={query.topic_id}: {candidate_cache}"
        )
    payload = json.loads(candidate_cache.read_text(encoding="utf-8"))
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"cache file missing response object: {candidate_cache}")
    return normalize_retrieved_candidates(response, query=query, retriever_name=retriever.name)


def _queries_by_topic(config: PipelineConfig) -> dict[str, QueryVariant]:
    topics = load_topics(config.topics.path, topic_format=config.topics.format)
    variant_configs = [{"name": variant.name, "type": variant.type} for variant in config.query_variants]
    queries: dict[str, QueryVariant] = {}
    for topic in topics:
        variants = build_query_variants(topic, variant_configs=variant_configs)
        if len(variants) != 1:
            raise ValueError("rerank score caching currently expects one query variant per topic")
        queries[topic.id] = variants[0]
    return queries


def _topics(config: PipelineConfig, requested_topic_ids: Sequence[str]) -> list[Topic]:
    topics = load_topics(config.topics.path, topic_format=config.topics.format)
    if requested_topic_ids:
        requested = set(requested_topic_ids)
        topics = [topic for topic in topics if topic.id in requested]
        missing = sorted(requested - {topic.id for topic in topics}, key=_topic_sort_key)
        if missing:
            raise ValueError(f"unknown topic ids: {', '.join(missing)}")
    return topics


def _topic_candidates(
    *,
    config: PipelineConfig,
    topic: Topic,
    query: QueryVariant,
    retriever: RetrieverConfig,
    cache_dir: Path,
    index_url: str,
    limit: int | None,
) -> list[RetrievedCandidate]:
    candidates = _load_cached_candidates(
        query=query,
        retriever=retriever,
        cache_dir=cache_dir,
        index_url=index_url,
    )
    ranked = passthrough_rank(candidates)
    selected = [row for row in ranked if row.topic_id == topic.id]
    if limit is not None:
        selected = selected[:limit]
    return [
        RetrievedCandidate(
            topic_id=row.topic_id,
            variant_name=query.variant_name,
            retriever_name=retriever.name,
            query_text=query.query_text,
            docid=row.docid,
            rank=row.rank,
            score=row.score,
            text=row.text,
        )
        for row in selected
    ]


def _scores_to_list(scores: Any) -> list[float]:
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, float | int):
        return [float(scores)]
    return [float(score) for score in scores]


def _predict(model: Any, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]:
    if not pairs:
        return []
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    if hasattr(scores, "detach"):
        scores = scores.detach().float().cpu()
    return _scores_to_list(scores)


def _choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_cross_encoder(model_name: str, *, max_length: int, device: str) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for reranker score caching. "
            "Example CPU run: uv run --with sentence-transformers "
            "python -m trec_rag.rerank_score_cache --topics 14"
        ) from exc
    return CrossEncoder(model_name, max_length=max_length, device=device)


def _score_document_rows(
    *,
    model: Any,
    topic: Topic,
    candidates: list[RetrievedCandidate],
    existing_keys: set[tuple[str, str]],
    batch_size: int,
    score_cache: GlobalScoreCache,
    score_kind: str,
) -> list[dict[str, Any]]:
    missing = [candidate for candidate in candidates if (topic.id, candidate.docid) not in existing_keys]
    rows: list[dict[str, Any]] = []
    pending: list[RetrievedCandidate] = []
    cache_hits = 0
    for candidate in missing:
        score = score_cache.get(query_text=candidate.query_text, text=candidate.text)
        if score is None:
            pending.append(candidate)
            continue
        cache_hits += 1
        existing_keys.add((topic.id, candidate.docid))
        rows.append(
            {
                "topic_id": topic.id,
                "docid": candidate.docid,
                "rank": candidate.rank,
                "score": score,
                "score_kind": score_kind,
                "score_cache_key": score_cache.cache_key(
                    query_text=candidate.query_text,
                    text=candidate.text,
                ),
            }
        )
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        scores = _predict(
            model,
            [(candidate.query_text, candidate.text) for candidate in batch],
            batch_size=batch_size,
        )
        score_cache.add_many(
            (candidate.query_text, candidate.text, score)
            for candidate, score in zip(batch, scores, strict=True)
        )
        for candidate, score in zip(batch, scores, strict=True):
            existing_keys.add((topic.id, candidate.docid))
            rows.append(
                {
                    "topic_id": topic.id,
                    "docid": candidate.docid,
                    "rank": candidate.rank,
                    "score": score,
                    "score_kind": score_kind,
                    "score_cache_key": score_cache.cache_key(
                        query_text=candidate.query_text,
                        text=candidate.text,
                    ),
                }
            )
    print(
        f"  document_global_cache_hits={cache_hits} document_model_scores={len(pending)}",
        flush=True,
    )
    return rows


def _score_window_rows(
    *,
    model: Any,
    topic: Topic,
    candidates: list[RetrievedCandidate],
    existing_docids: set[tuple[str, str]],
    batch_size: int,
    chunker: SemanticTextChunker,
    score_cache: GlobalScoreCache,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending: list[tuple[RetrievedCandidate, Any, int]] = []
    cache_hits = 0
    for candidate in candidates:
        if (topic.id, candidate.docid) in existing_docids:
            continue
        chunks = chunker.split_text(candidate.text, document_id=candidate.docid)
        for chunk_index, chunk in enumerate(chunks):
            score = score_cache.get(query_text=candidate.query_text, text=chunk.text)
            if score is None:
                pending.append((candidate, chunk, chunk_index))
                continue
            cache_hits += 1
            rows.append(
                {
                    "topic_id": topic.id,
                    "docid": candidate.docid,
                    "rank": candidate.rank,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk.chunk_id,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "score": score,
                    "score_cache_key": score_cache.cache_key(
                        query_text=candidate.query_text,
                        text=chunk.text,
                    ),
                }
            )
        existing_docids.add((topic.id, candidate.docid))

    pairs = [(candidate.query_text, chunk.text) for candidate, chunk, _ in pending]
    scores = _predict(model, pairs, batch_size=batch_size)
    score_cache.add_many(
        (candidate.query_text, chunk.text, score)
        for (candidate, chunk, _), score in zip(pending, scores, strict=True)
    )
    for (candidate, chunk, chunk_index), score in zip(pending, scores, strict=True):
        rows.append(
            {
                "topic_id": topic.id,
                "docid": candidate.docid,
                "rank": candidate.rank,
                "chunk_index": chunk_index,
                "chunk_id": chunk.chunk_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "score": score,
                "score_cache_key": score_cache.cache_key(
                    query_text=candidate.query_text,
                    text=chunk.text,
                ),
            }
        )
    print(
        f"  window_global_cache_hits={cache_hits} window_model_scores={len(pending)}",
        flush=True,
    )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build resumable Mixedbread reranker score JSONL artifacts from cached BM25 candidates."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/rag25_bm25_mixedbread_rerank_v1.yaml"))
    parser.add_argument("--topics", nargs="*", default=[], help="Optional topic ids to score.")
    parser.add_argument("--limit-per-topic", type=int, default=None)
    parser.add_argument("--score-kind", choices=["document", "window", "both"], default="both")
    parser.add_argument("--document-score-path", type=Path, default=None)
    parser.add_argument("--window-score-path", type=Path, default=None)
    parser.add_argument("--document-max-length", type=int, default=32768)
    parser.add_argument("--window-max-length", type=int, default=1024)
    parser.add_argument("--chunk-max-characters", type=int, default=3500)
    parser.add_argument("--chunk-overlap-characters", type=int, default=350)
    parser.add_argument("--document-batch-size", type=int, default=1)
    parser.add_argument("--window-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, etc. PyTorch ROCm uses cuda.")
    parser.add_argument("--sleep-between-topics", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_pipeline_config(args.config)
    reranker = config.ranking.reranker
    if not reranker:
        raise ValueError("config must use a cached-artifact reranker")
    if len(config.retrievers) != 1:
        raise ValueError("reranker score caching currently expects one retriever")
    if args.limit_per_topic is not None and args.limit_per_topic <= 0:
        raise ValueError("--limit-per-topic must be positive")

    load_repo_env(config.root_dir)
    os.environ.setdefault("INDEX_URL", DEFAULT_INDEX_URL)
    index_url = RemotePyseriniConfig.from_env().index_url
    cache_dir = pipeline_cache_dir(config.root_dir, config.run_id)
    retriever = config.retrievers[0]
    topics = _topics(config, args.topics)
    queries = _queries_by_topic(config)

    document_score_path = shared_output_path(
        config.root_dir,
        args.document_score_path or reranker.document_score_path,
    )
    window_score_path = shared_output_path(
        config.root_dir,
        args.window_score_path or reranker.window_score_path,
    )
    existing_document_keys = _read_jsonl_keys(document_score_path, ("topic_id", "docid"))
    existing_window_docids = _read_jsonl_keys(window_score_path, ("topic_id", "docid"))
    existing_document_scores = _read_document_scores(document_score_path)
    existing_window_scores = _read_window_scores(window_score_path)
    score_cache_root = global_score_cache_dir(config.root_dir)
    model_name = reranker.model
    document_score_kind = f"doc_max_{args.document_max_length}_buf512"
    document_score_cache = GlobalScoreCache(
        score_cache_root,
        ScoreCacheContext(
            backend="sentence-transformers-cross-encoder",
            model=model_name,
            max_length=args.document_max_length,
            score_kind=document_score_kind,
        ),
    )
    window_score_cache = GlobalScoreCache(
        score_cache_root,
        ScoreCacheContext(
            backend="sentence-transformers-cross-encoder",
            model=model_name,
            max_length=args.window_max_length,
            score_kind="window",
        ),
    )

    print(f"cache_dir={cache_dir}", flush=True)
    print(f"document_score_path={document_score_path}", flush=True)
    print(f"window_score_path={window_score_path}", flush=True)
    print(f"global_document_score_cache={document_score_cache.path}", flush=True)
    print(f"global_window_score_cache={window_score_cache.path}", flush=True)
    print(f"topics={','.join(topic.id for topic in topics)}", flush=True)
    print(f"index_url={index_url}", flush=True)
    if args.dry_run:
        for topic in topics:
            candidates = _topic_candidates(
                config=config,
                topic=topic,
                query=queries[topic.id],
                retriever=retriever,
                cache_dir=cache_dir,
                index_url=index_url,
                limit=args.limit_per_topic,
            )
            missing_docs = sum((topic.id, candidate.docid) not in existing_document_keys for candidate in candidates)
            missing_windows = sum((topic.id, candidate.docid) not in existing_window_docids for candidate in candidates)
            global_document_hits = sum(
                (topic.id, candidate.docid) not in existing_document_keys
                and document_score_cache.get(query_text=candidate.query_text, text=candidate.text) is not None
                for candidate in candidates
            )
            print(
                f"DRY topic={topic.id} candidates={len(candidates)} "
                f"missing_document_scores={missing_docs} "
                f"global_document_hits={global_document_hits} "
                f"missing_window_docs={missing_windows}",
                flush=True,
            )
        return 0

    device = _choose_device(args.device)
    print(f"model={model_name} device={device}", flush=True)

    document_model = None
    window_model = None
    chunker = SemanticTextChunker(
        ChunkingConfig(
            max_characters=args.chunk_max_characters,
            overlap_characters=args.chunk_overlap_characters,
        )
    )
    try:
        if args.score_kind in {"document", "both"}:
            document_model = _load_cross_encoder(
                model_name,
                max_length=args.document_max_length,
                device=device,
            )
        if args.score_kind in {"window", "both"}:
            window_model = _load_cross_encoder(
                model_name,
                max_length=args.window_max_length,
                device=device,
            )

        for topic_index, topic in enumerate(topics, start=1):
            query = queries[topic.id]
            candidates = _topic_candidates(
                config=config,
                topic=topic,
                query=query,
                retriever=retriever,
                cache_dir=cache_dir,
                index_url=index_url,
                limit=args.limit_per_topic,
            )
            print(
                f"TOPIC {topic.id} ({topic_index}/{len(topics)}) candidates={len(candidates)}",
                flush=True,
            )
            if args.score_kind in {"document", "both"}:
                seeded = _seed_document_score_cache(
                    topic=topic,
                    candidates=candidates,
                    existing_scores=existing_document_scores,
                    score_cache=document_score_cache,
                )
                if seeded:
                    print(f"  document_global_cache_seeded={seeded}", flush=True)
            if args.score_kind in {"window", "both"}:
                seeded = _seed_window_score_cache(
                    topic=topic,
                    candidates=candidates,
                    existing_scores=existing_window_scores,
                    score_cache=window_score_cache,
                    chunker=chunker,
                )
                if seeded:
                    print(f"  window_global_cache_seeded={seeded}", flush=True)
            if document_model is not None:
                rows = _score_document_rows(
                    model=document_model,
                    topic=topic,
                    candidates=candidates,
                    existing_keys=existing_document_keys,
                    batch_size=args.document_batch_size,
                    score_cache=document_score_cache,
                    score_kind=document_score_kind,
                )
                written = _append_jsonl(document_score_path, rows)
                print(f"  document_scores_written={written}", flush=True)
            if window_model is not None:
                rows = _score_window_rows(
                    model=window_model,
                    topic=topic,
                    candidates=candidates,
                    existing_docids=existing_window_docids,
                    batch_size=args.window_batch_size,
                    chunker=chunker,
                    score_cache=window_score_cache,
                )
                written = _append_jsonl(window_score_path, rows)
                print(f"  window_scores_written={written}", flush=True)
            if topic_index < len(topics):
                time.sleep(args.sleep_between_topics)
    finally:
        del document_model
        del window_model
        gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
