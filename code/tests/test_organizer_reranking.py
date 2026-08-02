from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag.organizer_reranking import (
    LISTWISE_SCHEMA,
    POINTWISE_SCHEMA,
    QWEN_RERANKER_SUFFIX,
    RerankCandidate,
    first_label_token_ids,
    first_permutation_from_logprobs,
    first_prompt_token_ids,
    first_sliding_window_rerank,
    load_bm25_candidates,
    load_listwise_order,
    load_pointwise_scores,
    pointwise_score_row,
    qwen_label_token_ids,
    qwen_pointwise_token_ids,
    retrieval_document_text,
    retrieval_query_text,
    sha256_canonical_text,
    sliding_window_bounds,
    validate_pointwise_score_identity,
)


class FakeTokenizer:
    def __init__(self) -> None:
        labels = ["yes", "no", *(chr(ord("A") + index) for index in range(26))]
        self.labels = {label: index + 1 for index, label in enumerate(labels)}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.labels[token]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if text in self.labels:
            return [self.labels[text]]
        return [1000 + ord(character) for character in text]

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> list[int]:
        rendered = "\n".join(message["content"] for message in conversation) + "\nassistant:"
        return self.encode(rendered)


class MappingChatTemplateTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> dict[str, list[list[int]]]:
        token_ids = super().apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        return {"input_ids": [token_ids]}


def candidate(rank: int, *, topic: str = "1") -> RerankCandidate:
    return RerankCandidate(topic, "query", f"d{rank}", rank, 100.0 - rank, f"document {rank}")


def test_retrieval_query_text_accepts_api_mapping_without_stringifying_it() -> None:
    narrative = "A narrative query"

    assert retrieval_query_text({"query": narrative}) == narrative
    assert retrieval_query_text({"query": {"text": narrative}}) == narrative
    with pytest.raises(ValueError, match="must be text"):
        retrieval_query_text({"query": {"value": narrative}})


def test_canonical_text_hash_ignores_layout_whitespace_only() -> None:
    assert sha256_canonical_text("one\n\ntwo") == sha256_canonical_text("one two")
    assert sha256_canonical_text("one two") != sha256_canonical_text("one three")


def test_retrieval_document_text_decodes_supported_api_shapes() -> None:
    assert retrieval_document_text("plain text") == "plain text"
    assert retrieval_document_text({"text": "mapped text"}) == "mapped text"
    assert retrieval_document_text('["first sentence", "second sentence"]') == (
        "first sentence second sentence"
    )
    with pytest.raises(ValueError, match="no text field"):
        retrieval_document_text({"title": "not body text"})


def test_qwen_prompt_preserves_suffix_when_document_is_truncated() -> None:
    tokenizer = FakeTokenizer()
    token_ids = qwen_pointwise_token_ids(
        tokenizer,
        query="short query",
        document="x" * 1000,
        max_length=500,
    )

    assert len(token_ids) == 500
    assert token_ids[-len(tokenizer.encode(QWEN_RERANKER_SUFFIX)) :] == tokenizer.encode(
        QWEN_RERANKER_SUFFIX
    )
    assert qwen_label_token_ids(tokenizer) == (1, 2)


def test_first_prompt_and_logits_use_all_alphabetical_labels() -> None:
    tokenizer = FakeTokenizer()
    token_ids, words = first_prompt_token_ids(
        tokenizer,
        query="query",
        documents=["one two three", "four five six"],
        context_size=900,
        max_passage_words=3,
    )
    labels = first_label_token_ids(tokenizer, 2)

    assert token_ids
    assert words >= 1
    assert first_permutation_from_logprobs(labels, {labels[0]: -2.0, labels[1]: -0.5}) == [1, 0]


def test_first_prompt_accepts_transformers_mapping_result() -> None:
    token_ids, words = first_prompt_token_ids(
        MappingChatTemplateTokenizer(),
        query="query",
        documents=["one two three", "four five six"],
        context_size=900,
        max_passage_words=3,
    )

    assert token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert words == 3


def test_first_sliding_windows_match_rankllm_tail_to_head_schedule() -> None:
    assert sliding_window_bounds(100, window_size=20, stride=10) == [
        (80, 100),
        (70, 90),
        (60, 80),
        (50, 70),
        (40, 60),
        (30, 50),
        (20, 40),
        (10, 30),
        (0, 20),
    ]
    calls: list[list[str]] = []

    def reverse_window(_query: str, rows: tuple[RerankCandidate, ...]) -> list[int]:
        calls.append([row.docid for row in rows])
        return list(reversed(range(len(rows))))

    reranked = first_sliding_window_rerank(
        [candidate(rank) for rank in range(1, 101)],
        window_size=20,
        stride=10,
        rank_window=reverse_window,
    )

    assert len(calls) == 9
    assert len(reranked) == 100
    assert {row.docid for row in reranked} == {f"d{rank}" for rank in range(1, 101)}


def test_candidate_and_inference_artifact_loaders_are_strict(tmp_path: Path) -> None:
    bm25_path = tmp_path / "bm25.jsonl"
    bm25_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "topic_id": "1",
                    "variant_name": "original",
                    "query_text": "query",
                    "docid": f"d{rank}",
                    "rank": rank,
                    "score": 10 - rank,
                    "text": f"document {rank}",
                }
            )
            for rank in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(load_bm25_candidates(bm25_path, expected_depth=2)["1"]) == 2

    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        json.dumps(
            {
                "schema_version": POINTWISE_SCHEMA,
                "topic_id": "1",
                "docid": "d1",
                "score": 0.75,
                "model": "model",
                "dtype": "bfloat16",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_pointwise_scores(scores_path, expected_model="model") == {("1", "d1"): 0.75}

    order_path = tmp_path / "order.jsonl"
    order_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": LISTWISE_SCHEMA,
                    "topic_id": "1",
                    "docid": docid,
                    "rank": rank,
                }
            )
            for rank, docid in ((1, "d2"), (2, "d1"))
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_listwise_order(order_path) == {"1": ["d2", "d1"]}


def test_pointwise_identity_validation_checks_candidate_text(tmp_path: Path) -> None:
    row = candidate(1)
    score_row = pointwise_score_row(
        row,
        score=0.75,
        model="model",
        model_revision="revision",
        dtype="bfloat16",
        max_length=8192,
        prompt_tokens=100,
        backend="test",
    )
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(score_row) + "\n", encoding="utf-8")

    validate_pointwise_score_identity(path, {"1": [row]})

    score_row["text_sha256"] = "0" * 64
    path.write_text(json.dumps(score_row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="document text identity mismatch"):
        validate_pointwise_score_identity(path, {"1": [row]})


def test_first_rejects_incomplete_label_logits() -> None:
    tokenizer = FakeTokenizer()
    labels = first_label_token_ids(tokenizer, 2)
    with pytest.raises(ValueError, match="missing label logits"):
        first_permutation_from_logprobs(labels, {labels[0]: -1.0})
