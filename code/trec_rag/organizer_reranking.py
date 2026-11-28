"""Organizer-style Qwen pointwise and FIRST listwise reranking helpers.

The functions in this module deliberately keep inference backends at the
boundary.  Prompt construction, candidate validation, sliding windows, and
artifact schemas are shared by the local vLLM and Modal runners.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


POINTWISE_SCHEMA = "qwen3-pointwise-scores-v1"
LISTWISE_SCHEMA = "first-qwen3-listwise-run-v1"
LISTWISE_SEED_SCHEMA = "mixedbread-listwise-seed-v1"

QWEN_RERANKER_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
QWEN_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QWEN_RERANKER_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

FIRST_SYSTEM = (
    "You are RankLLM, an intelligent assistant that can rank passages based on "
    "their relevancy to the query"
)


class Tokenizer(Protocol):
    """The small tokenizer surface needed by both inference implementations."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> list[int] | str | Mapping[str, object]: ...


@dataclass(frozen=True)
class RerankCandidate:
    topic_id: str
    query_text: str
    docid: str
    bm25_rank: int
    bm25_score: float
    text: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_canonical_text(value: str) -> str:
    """Hash text after the same whitespace collapse used by FIRST passages."""

    return sha256_text(" ".join(value.split()))


def retrieval_query_text(payload: Mapping[str, object]) -> str:
    """Extract query text from either Pyserini cache response shape."""

    query = payload.get("query")
    if isinstance(query, str):
        text = query
    elif isinstance(query, Mapping) and isinstance(query.get("text"), str):
        text = str(query["text"])
    else:
        raise ValueError("retrieval response query must be text or contain a text field")
    if not text.strip():
        raise ValueError("retrieval response query is empty")
    return text


def retrieval_document_text(value: object) -> str:
    """Normalize document payloads emitted by the ClimbMix search API."""

    if isinstance(value, Mapping):
        for key in ("text", "contents", "body"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        raise ValueError("retrieval document mapping has no text field")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("retrieval document is empty or not text")
    text = value.strip()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(decoded, list) and decoded and all(isinstance(item, str) for item in decoded):
        joined = " ".join(item for item in decoded if item)
        if joined.strip():
            return joined.strip()
    if isinstance(decoded, str) and decoded.strip():
        return decoded.strip()
    if isinstance(decoded, Mapping):
        return retrieval_document_text(decoded)
    return text


def load_bm25_candidates(
    path: Path,
    *,
    expected_depth: int | None = None,
) -> dict[str, list[RerankCandidate]]:
    """Load and strictly validate one original-query BM25 stream per topic."""

    by_topic: dict[str, list[RerankCandidate]] = {}
    seen: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                candidate = RerankCandidate(
                    topic_id=str(row["topic_id"]),
                    query_text=str(row["query_text"]),
                    docid=str(row["docid"]),
                    bm25_rank=int(row["rank"]),
                    bm25_score=float(row["score"]),
                    text=str(row["text"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid BM25 candidate") from exc
            if row.get("variant_name") != "original":
                raise ValueError(f"{path}:{line_number}: expected the original query stream")
            if not candidate.docid or not candidate.query_text or not candidate.text:
                raise ValueError(f"{path}:{line_number}: candidate text fields are empty")
            if candidate.bm25_rank < 1 or not math.isfinite(candidate.bm25_score):
                raise ValueError(f"{path}:{line_number}: invalid BM25 rank or score")
            key = (candidate.topic_id, candidate.docid)
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate topic-document pair {key}")
            seen.add(key)
            by_topic.setdefault(candidate.topic_id, []).append(candidate)

    if not by_topic:
        raise ValueError(f"{path}: no BM25 candidates")
    for topic_id, rows in by_topic.items():
        rows.sort(key=lambda row: row.bm25_rank)
        if [row.bm25_rank for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"topic {topic_id}: BM25 ranks are not contiguous")
        if len({row.query_text for row in rows}) != 1:
            raise ValueError(f"topic {topic_id}: multiple original query texts")
        if expected_depth is not None and len(rows) != expected_depth:
            raise ValueError(
                f"topic {topic_id}: expected {expected_depth} candidates; found {len(rows)}"
            )
    return dict(sorted(by_topic.items(), key=lambda item: item[0]))


def load_bm25_candidate_sample(
    path: Path,
    *,
    topic_id: str,
    ranks: Sequence[int],
    expected_depth: int,
) -> list[RerankCandidate]:
    """Stream a small rank sample while validating the topic's complete depth."""

    requested = {int(rank) for rank in ranks}
    if not requested or any(rank < 1 for rank in requested):
        raise ValueError("sample ranks must be positive")
    selected: dict[int, RerankCandidate] = {}
    observed_ranks: set[int] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("topic_id")) != topic_id:
                continue
            try:
                candidate = RerankCandidate(
                    topic_id=topic_id,
                    query_text=str(row["query_text"]),
                    docid=str(row["docid"]),
                    bm25_rank=int(row["rank"]),
                    bm25_score=float(row["score"]),
                    text=str(row["text"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid BM25 candidate") from exc
            if row.get("variant_name") != "original" or candidate.bm25_rank in observed_ranks:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate sample stream row")
            observed_ranks.add(candidate.bm25_rank)
            if candidate.bm25_rank in requested:
                selected[candidate.bm25_rank] = candidate
    if observed_ranks != set(range(1, expected_depth + 1)):
        raise ValueError(
            f"topic {topic_id}: expected contiguous depth {expected_depth}; "
            f"found {len(observed_ranks)} rows"
        )
    missing = requested - set(selected)
    if missing:
        raise ValueError(f"topic {topic_id}: sample ranks are missing: {sorted(missing)}")
    rows = [selected[int(rank)] for rank in ranks]
    if len({row.query_text for row in rows}) != 1:
        raise ValueError(f"topic {topic_id}: sample contains multiple query texts")
    return rows


def qwen_label_token_ids(tokenizer: Tokenizer) -> tuple[int, int]:
    """Return the official positive and negative single-token identifiers."""

    yes_id = int(tokenizer.convert_tokens_to_ids("yes"))
    no_id = int(tokenizer.convert_tokens_to_ids("no"))
    if yes_id < 0 or no_id < 0 or yes_id == no_id:
        raise ValueError("Qwen yes/no label tokens are invalid")
    if tokenizer.encode("yes", add_special_tokens=False) != [yes_id]:
        raise ValueError("Qwen 'yes' label is not a single token")
    if tokenizer.encode("no", add_special_tokens=False) != [no_id]:
        raise ValueError("Qwen 'no' label is not a single token")
    return yes_id, no_id


def qwen_pointwise_token_ids(
    tokenizer: Tokenizer,
    *,
    query: str,
    document: str,
    max_length: int,
    instruction: str = QWEN_RERANKER_INSTRUCTION,
) -> list[int]:
    """Build the exact Qwen3-Reranker prompt while preserving its suffix."""

    if max_length < 32:
        raise ValueError("pointwise max_length is too small")
    prefix = (
        f"<|im_start|>system\n{QWEN_RERANKER_SYSTEM}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: "
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(QWEN_RERANKER_SUFFIX, add_special_tokens=False)
    document_ids = tokenizer.encode(document, add_special_tokens=False)
    document_budget = max_length - len(prefix_ids) - len(suffix_ids)
    if document_budget < 1:
        raise ValueError("query and fixed prompt exceed pointwise max_length")
    return [*prefix_ids, *document_ids[:document_budget], *suffix_ids]


def qwen_pointwise_batch_token_ids(
    tokenizer: Tokenizer,
    *,
    query: str,
    documents: Sequence[str],
    max_length: int,
    instruction: str = QWEN_RERANKER_INSTRUCTION,
) -> list[list[int]]:
    return [
        qwen_pointwise_token_ids(
            tokenizer,
            query=query,
            document=document,
            max_length=max_length,
            instruction=instruction,
        )
        for document in documents
    ]


def first_messages(
    *,
    query: str,
    documents: Sequence[str],
    max_passage_words: int,
) -> list[dict[str, str]]:
    """Render RankLLM's released alphabetical FIRST prompt template."""

    if not documents or len(documents) > 26:
        raise ValueError("FIRST requires one through 26 passages")
    if max_passage_words < 1:
        raise ValueError("max_passage_words must be positive")
    count = len(documents)
    prefix = (
        f"I will provide you with {count} passages, each indicated by an "
        "alphabetical identifier []. Rank the passages based on their relevance "
        f"to the search query: {query}.\n"
    )
    body = "\n".join(
        f"[{chr(ord('A') + index)}] {' '.join(document.split()[:max_passage_words])}"
        for index, document in enumerate(documents)
    )
    suffix = (
        f"Search Query: {query}.\nRank the {count} passages above based on their "
        "relevance to the search query. All the passages should be included and "
        "listed using identifiers, in descending order of relevance. The output "
        "format should be [] > [], e.g., [B] > [A], Answer concisely and directly "
        "and only respond with the ranking results, do not say any word or explain."
    )
    return [
        {"role": "system", "content": FIRST_SYSTEM},
        {"role": "user", "content": f"{prefix}{body}\n{suffix}"},
    ]


def first_label_token_ids(tokenizer: Tokenizer, count: int) -> list[int]:
    if count < 1 or count > 26:
        raise ValueError("FIRST label count must be from one through 26")
    result: list[int] = []
    for index in range(count):
        label = chr(ord("A") + index)
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"FIRST label {label!r} is not a single token")
        result.append(int(encoded[0]))
    if len(set(result)) != count:
        raise ValueError("FIRST label token ids are not unique")
    return result


def first_prompt_token_ids(
    tokenizer: Tokenizer,
    *,
    query: str,
    documents: Sequence[str],
    context_size: int,
    max_passage_words: int,
) -> tuple[list[int], int]:
    """Render a FIRST prompt and shrink passages using RankLLM's policy."""

    labels = " > ".join(f"[{chr(ord('A') + index)}]" for index in range(len(documents)))
    output_reserve = len(tokenizer.encode(labels, add_special_tokens=False))
    current_words = max_passage_words
    while current_words > 0:
        messages = first_messages(
            query=query,
            documents=documents,
            max_passage_words=current_words,
        )
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        if isinstance(rendered, Mapping):
            if "input_ids" not in rendered:
                raise ValueError("FIRST chat template result is missing input_ids")
            rendered = rendered["input_ids"]
        to_list = getattr(rendered, "tolist", None)
        if callable(to_list):
            rendered = to_list()
        if not isinstance(rendered, (list, tuple)):
            raise ValueError("FIRST chat template did not return token ids")
        if rendered and isinstance(rendered[0], (list, tuple)):
            if len(rendered) != 1:
                raise ValueError("FIRST chat template returned multiple prompt batches")
            rendered = rendered[0]
        token_ids = [int(token_id) for token_id in rendered]
        overflow = len(token_ids) + output_reserve - context_size
        if overflow <= 0:
            return token_ids, current_words
        current_words -= max(1, overflow // (len(documents) * 4))
    raise ValueError("FIRST prompt does not fit the configured context")


def first_permutation_from_logprobs(
    label_token_ids: Sequence[int],
    logprobs: Mapping[int, float],
) -> list[int]:
    """Convert first-token label log-probabilities into a stable permutation."""

    missing = [token_id for token_id in label_token_ids if token_id not in logprobs]
    if missing:
        raise ValueError(f"FIRST output is missing label logits: {missing}")
    return sorted(
        range(len(label_token_ids)),
        key=lambda index: (-float(logprobs[label_token_ids[index]]), index),
    )


def sliding_window_bounds(
    candidate_count: int,
    *,
    window_size: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Return RankLLM's tail-to-head sliding-window schedule."""

    if candidate_count < 1 or window_size < 1 or stride < 1 or stride > window_size:
        raise ValueError("invalid listwise window configuration")
    end = candidate_count
    start = max(end - window_size, 0)
    bounds: list[tuple[int, int]] = []
    previous_start: int | None = None
    while end > start and previous_start != 0:
        bounds.append((start, end))
        previous_start = start
        end -= stride
        start = max(start - stride, 0)
    return bounds


def first_sliding_window_rerank(
    candidates: Sequence[RerankCandidate],
    *,
    window_size: int,
    stride: int,
    rank_window: Callable[[str, Sequence[RerankCandidate]], Sequence[int]],
) -> list[RerankCandidate]:
    """Apply validated FIRST window permutations from the tail toward rank one."""

    if not candidates:
        return []
    query_texts = {candidate.query_text for candidate in candidates}
    if len(query_texts) != 1:
        raise ValueError("listwise candidates must share one query")
    query = next(iter(query_texts))
    reranked = list(candidates)
    for start, end in sliding_window_bounds(
        len(reranked), window_size=window_size, stride=stride
    ):
        window = reranked[start:end]
        permutation = [int(index) for index in rank_window(query, tuple(window))]
        if sorted(permutation) != list(range(len(window))):
            raise ValueError(f"FIRST window {start}:{end} did not return a permutation")
        reranked[start:end] = [window[index] for index in permutation]
    return reranked


def iter_batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_pointwise_scores(
    path: Path,
    *,
    expected_model: str | None = None,
    expected_dtype: str = "bfloat16",
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != POINTWISE_SCHEMA:
                raise ValueError(f"{path}:{line_number}: pointwise schema mismatch")
            if expected_model is not None and row.get("model") != expected_model:
                raise ValueError(f"{path}:{line_number}: pointwise model mismatch")
            if row.get("dtype") != expected_dtype:
                raise ValueError(f"{path}:{line_number}: pointwise dtype mismatch")
            key = (str(row["topic_id"]), str(row["docid"]))
            score = float(row["score"])
            if key in scores or not 0.0 <= score <= 1.0 or not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: duplicate or invalid score")
            scores[key] = score
    return scores


def validate_pointwise_score_identity(
    path: Path,
    candidates: Mapping[str, Sequence[RerankCandidate]],
) -> None:
    """Prove that a pointwise artifact scores the exact candidate text pool."""

    expected = {
        (topic_id, candidate.docid): candidate
        for topic_id, rows in candidates.items()
        for candidate in rows
    }
    seen: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("topic_id")), str(row.get("docid")))
            candidate = expected.get(key)
            if candidate is None or key in seen:
                raise ValueError(f"{path}:{line_number}: unexpected or duplicate candidate identity")
            if int(row.get("bm25_rank", 0)) != candidate.bm25_rank:
                raise ValueError(f"{path}:{line_number}: BM25 rank identity mismatch")
            if row.get("query_sha256") != sha256_text(candidate.query_text):
                raise ValueError(f"{path}:{line_number}: query identity mismatch")
            if row.get("text_sha256") != sha256_text(candidate.text):
                raise ValueError(f"{path}:{line_number}: document text identity mismatch")
            seen.add(key)
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"{path}: pointwise artifact is missing {len(missing)} candidates")


def load_listwise_order(
    path: Path,
    *,
    expected_model: str | None = None,
    expected_dtype: str | None = None,
) -> dict[str, list[str]]:
    by_topic: dict[str, list[tuple[int, str]]] = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != LISTWISE_SCHEMA:
                raise ValueError(f"{path}:{line_number}: listwise schema mismatch")
            if expected_model is not None and row.get("model") != expected_model:
                raise ValueError(f"{path}:{line_number}: listwise model mismatch")
            if expected_dtype is not None and row.get("dtype") != expected_dtype:
                raise ValueError(f"{path}:{line_number}: listwise dtype mismatch")
            topic_id = str(row["topic_id"])
            by_topic.setdefault(topic_id, []).append((int(row["rank"]), str(row["docid"])))
    result: dict[str, list[str]] = {}
    for topic_id, rows in by_topic.items():
        rows.sort()
        if [rank for rank, _docid in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"topic {topic_id}: listwise ranks are not contiguous")
        docids = [docid for _rank, docid in rows]
        if len(set(docids)) != len(docids):
            raise ValueError(f"topic {topic_id}: duplicate listwise documents")
        result[topic_id] = docids
    return result


def pointwise_score_row(
    candidate: RerankCandidate,
    *,
    score: float,
    model: str,
    model_revision: str,
    dtype: str,
    max_length: int,
    prompt_tokens: int,
    backend: str,
) -> dict[str, object]:
    if not 0.0 <= score <= 1.0 or not math.isfinite(score):
        raise ValueError("pointwise probability must be finite and in [0, 1]")
    return {
        "schema_version": POINTWISE_SCHEMA,
        "topic_id": candidate.topic_id,
        "docid": candidate.docid,
        "bm25_rank": candidate.bm25_rank,
        "bm25_score": candidate.bm25_score,
        "score": score,
        "score_kind": "yes_probability_over_yes_no",
        "query_sha256": sha256_text(candidate.query_text),
        "text_sha256": sha256_text(candidate.text),
        "prompt_tokens": prompt_tokens,
        "model": model,
        "model_revision": model_revision,
        "dtype": dtype,
        "max_length": max_length,
        "backend": backend,
    }


def write_jsonl_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
