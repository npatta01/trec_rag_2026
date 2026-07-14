from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import json
import pytest

from trec_rag.adaptive_evidence_discovery import (
    attach_contract_folds,
    build_discovery_messages,
    build_derived_query,
    extract_repeated_phrases,
    freeze_o1,
    finalize_discovery_records,
    build_reservoirs,
    merge_nuggets,
    qualify_opposite_support,
    record_discovery_unavailable,
    run_discovery_preflight,
    run_proposal_pass,
    validate_model_response,
    validate_proposal,
    verify_discovery_terminal,
)
from trec_rag.adaptive_evidence_local_model import (
    DISCOVERY_SCHEMA,
    MODEL_REVISION,
    LocalJsonModel,
)


def _o0() -> list[dict[str, object]]:
    return [
        {
            "topic_id": "219",
            "obligation_id": "219-positive",
            "kind": "o0",
            "text": "Positive effects of technology on society and daily life.",
            "anchor_terms": ["technology"],
            "relation_terms": ["positive", "society", "daily life"],
        }
    ]


def _passages(count: int) -> list[dict[str, object]]:
    return [
        {
            "topic_id": "219",
            "variant": "219-positive",
            "document_id": f"doc-{index // 2:02d}",
            "fold": (index // 2) % 2,
            "score": 100.0 - index,
            "window_id": f"window-{index:02d}",
            "window_text": f"technology produces positive social effect {index}",
        }
        for index in range(count * 2)
    ]


def _parent() -> dict[str, object]:
    return {
        "topic_id": "219",
        "obligation_id": "219-positive",
        "kind": "o0",
        "text": "Positive effects of technology on society and daily life.",
        "anchor_terms": ["technology"],
        "relation_terms": ["positive", "society", "daily life"],
    }


def _proposal(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "proposal_id": "219-positive:f0:0",
        "topic_id": "219",
        "parent_id": "219-positive",
        "kind": "o1",
        "label": "accessibility benefits of technology",
        "scope_rationale": "A category of positive effects within the parent scope.",
        "document_id": "source-doc",
        "fold": 0,
        "support_span": "technology improves accessibility",
        "subject": "technology",
        "population": "society",
        "relation": "positive effects",
    }
    value.update(changes)
    return value


def _support() -> list[dict[str, object]]:
    return [
        {
            "document_id": "other-doc",
            "fold": 1,
            "qualified": True,
            "support_span": "technology improves accessibility",
        }
    ]


def _nugget(text: str) -> dict[str, object]:
    return {
        "topic_id": "219",
        "parent_id": "219-positive",
        "subject": "technology",
        "relation": "improves",
        "object": text,
        "support_span": text,
        "document_id": text,
        "fold": 0,
    }


def test_reservoir_has_ten_distinct_docs_per_parent_fold() -> None:
    rows = build_reservoirs(_o0(), _passages(30), limit=10)
    assert len(rows[("219-positive", 0)]) == 10
    assert len({row["document_id"] for row in rows[("219-positive", 0)]}) == 10


def test_authenticated_contract_folds_are_joined_to_base_score_rows() -> None:
    rows = [
        {
            "topic_id": "219",
            "document_id": "doc-1",
            "variant": "219-positive",
            "score": 1.0,
        }
    ]
    documents = [{"topic_id": "219", "document_id": "doc-1", "fold": 1}]
    joined = attach_contract_folds(rows, documents)
    assert joined[0]["fold"] == 1
    assert "fold" not in rows[0]


def test_answer_fact_cannot_become_o1() -> None:
    proposal = _proposal(label="COVID-19 caused a 42 percent increase", kind="o1")
    decision = validate_proposal(
        proposal,
        _parent(),
        opposite_fold_support=_support(),
    )
    assert decision["accepted"] is False
    assert "candidate_answer" in decision["reasons"]


def test_o1_requires_distinct_opposite_fold_document() -> None:
    decision = validate_proposal(
        _proposal(),
        _parent(),
        opposite_fold_support=[],
    )
    assert decision["accepted"] is False
    assert "cross_fold_support" in decision["reasons"]


def test_nugget_merging_preserves_singletons_and_merges_jaccard_080() -> None:
    merged = merge_nuggets(
        [
            _nugget("a b c d e f g h i"),
            _nugget("a b c d e f g h j"),
            _nugget("rare x"),
        ]
    )
    assert len(merged) == 2
    assert any(row["singleton"] is True for row in merged)


class _Batch(dict[str, object]):
    def __init__(self) -> None:
        super().__init__(input_ids=SimpleNamespace(shape=(1, 3)))
        self.input_ids = self["input_ids"]
        self.device: str | None = None

    def to(self, device: str) -> _Batch:
        self.device = device
        return self


class _FakeTokenizer:
    eos_token_id = 7

    def __init__(self) -> None:
        self.messages: object = None

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.messages = messages
        return "prompt"

    def __call__(self, prompt: str, **kwargs: object) -> _Batch:
        assert prompt == "prompt"
        assert kwargs == {"return_tensors": "pt"}
        return _Batch()

    def decode(self, tokens: object, **kwargs: object) -> str:
        assert tokens == [4, 5]
        assert kwargs == {"skip_special_tokens": True}
        return '{"n1":[],"o1":[],"status":"unsupported"}'


class _FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.device: str | None = None
        self.generate_kwargs: dict[str, object] = {}

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generate_kwargs = dict(kwargs)
        return [[1, 2, 3, 4, 5]]


class _FakeLoader:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[object, dict[str, object]]] = []

    def from_pretrained(self, path: object, **kwargs: object) -> object:
        self.calls.append((path, dict(kwargs)))
        return self.value


def test_local_json_model_is_pinned_bf16_eval_cuda_and_deterministic() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    tokenizer_loader = _FakeLoader(tokenizer)
    model_loader = _FakeLoader(model)
    seeded: list[int] = []
    runtime = SimpleNamespace(
        torch=SimpleNamespace(
            bfloat16="bf16",
            inference_mode=nullcontext,
            manual_seed=seeded.append,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda index: "fake AMD",
                manual_seed_all=seeded.append,
                reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 123,
            ),
            __version__="fake torch",
            version=SimpleNamespace(hip="fake hip"),
        ),
        auto_tokenizer_cls=tokenizer_loader,
        auto_model_cls=model_loader,
        clock=lambda: 1.0,
        host_memory_bytes=lambda: 456,
    )

    adapter = LocalJsonModel(runtime=runtime)
    messages = build_discovery_messages(
        _parent(),
        [
            {
                "document_id": "doc-1",
                "fold": 0,
                "passage_text": "Technology improves information access.",
            }
        ],
    )
    value = adapter.generate(messages, DISCOVERY_SCHEMA, max_new_tokens=99)

    assert value["status"] == "unsupported"
    assert tokenizer.messages == messages
    assert json.loads(messages[1]["content"])["response_json_schema"] == (
        DISCOVERY_SCHEMA
    )
    assert "response_json_schema is authoritative" in messages[0]["content"]
    assert tokenizer_loader.calls[0][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert model_loader.calls[0][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "torch_dtype": "bf16",
    }
    assert model.eval_called is True
    assert model.device == "cuda"
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 99
    assert model.generate_kwargs["pad_token_id"] == 7
    assert seeded == [0, 0]
    assert adapter.execution_receipt()["model_revision"] == MODEL_REVISION


def test_model_response_rejects_spans_not_in_supplied_passage() -> None:
    passages = [
        {
            "document_id": "doc-1",
            "fold": 0,
            "passage_text": "Technology can improve access to information.",
        }
    ]
    response = {
        "status": "supported",
        "o1": [
            {
                "label": "information access benefits",
                "scope_rationale": "A positive societal effect of technology.",
                "subject": "technology",
                "population": "society",
                "relation": "positive effects",
                "support_document_id": "doc-1",
                "support_span": "outside knowledge",
            }
        ],
        "n1": [],
    }

    accepted, rejected = validate_model_response(response, passages)

    assert accepted == {"o1": [], "n1": []}
    assert rejected[0]["reason"] == "support_span"


def test_repeated_phrase_control_requires_two_documents_across_folds() -> None:
    rows = extract_repeated_phrases(
        [
            {
                "document_id": "a",
                "fold": 0,
                "passage_text": "Technology improves daily information access.",
            },
            {
                "document_id": "b",
                "fold": 1,
                "passage_text": "Technology improves daily information access for families.",
            },
            {
                "document_id": "c",
                "fold": 0,
                "passage_text": "Technology improves daily information access.",
            },
        ]
    )
    by_phrase = {row["phrase"]: row for row in rows}
    assert by_phrase["technology improves"]["folds"] == [0, 1]
    assert by_phrase["technology improves"]["document_ids"] == ["a", "b", "c"]
    assert "families" not in by_phrase


def test_freeze_o1_is_lexicographic_and_caps_parent_and_topic() -> None:
    rows: list[dict[str, object]] = []
    for index in range(6):
        rows.append(
            {
                "accepted": True,
                "topic_id": "219",
                "parent_id": "219-positive" if index < 2 else f"219-p{index}",
                "label": "z label" if index == 0 else f"label {index}",
                "validating_document_count": 2,
                "independent_stream_count": 1,
                "source_diversity": 1,
                "parent_local_rank": index,
            }
        )

    accepted, rejected = freeze_o1(rows)

    assert len(accepted) == 4
    assert len({row["parent_id"] for row in accepted}) == 4
    assert all("freeze_limit" in row["reasons"] for row in rejected)


def test_derived_query_contains_full_narrative_parent_and_heading() -> None:
    broad = {"text": "full narrative"}
    parent = {"text": "complete parent O0"}
    query = build_derived_query(broad, parent, "information access")
    assert query == (
        "full narrative\n\nExplicit obligation:\ncomplete parent O0"
        "\n\nCorpus-derived sub-obligation:\ninformation access"
    )


def test_discovery_prompt_freezes_scope_spans_and_no_outside_knowledge() -> None:
    passages = [
        {
            "document_id": "doc-1",
            "fold": 0,
            "passage_text": "Technology improves information access.",
        }
    ]
    messages = build_discovery_messages(_parent(), passages)
    content = "\n".join(message["content"] for message in messages)
    assert "preserve the parent subject, population, domain, and relation" in content
    assert "abstract O1 categories separately from specific N1 facts" in content
    assert "exact substrings from the supplied passages" in content
    assert "never use outside knowledge" in content
    assert "return unsupported rather than inventing evidence" in content
    assert "Technology improves information access." in content


def test_discovery_prompt_embeds_exact_authoritative_frozen_schema() -> None:
    messages = build_discovery_messages(
        _parent(),
        [
            {
                "document_id": "doc-1",
                "fold": 0,
                "passage_text": "Technology improves information access.",
            }
        ],
    )
    payload = json.loads(messages[1]["content"])
    assert payload["response_json_schema"] == DISCOVERY_SCHEMA
    assert "response_json_schema is authoritative" in messages[0]["content"]
    assert "exactly one JSON object" in messages[0]["content"]


def test_deterministic_support_uses_exact_opposite_fold_anchor_relation_sentence() -> None:
    proposal = _proposal(
        label="information access benefits",
        source_fold=0,
        source_document_sha256="source-sha",
        obligation_id="219-positive:f0:o1:0",
    )
    scores = [
        {
            "variant": "219-positive:f0:o1:0",
            "document_id": "same-fold",
            "document_sha256": "same-fold-sha",
            "fold": 0,
            "score": 9.0,
            "window_id": "same",
            "window_text": "Technology has positive effects on society.",
        },
        {
            "variant": "219-positive:f0:o1:0",
            "document_id": "opposite",
            "document_sha256": "opposite-sha",
            "fold": 1,
            "score": 8.0,
            "window_id": "opposite-window",
            "window_text": (
                "Unrelated preface. Technology improves information access, "
                "a positive effect for society. Another sentence."
            ),
        },
    ]

    support = qualify_opposite_support(proposal, _parent(), scores)

    assert len(support) == 1
    assert support[0]["document_id"] == "opposite"
    assert support[0]["fold"] == 1
    assert support[0]["qualified"] is True
    assert support[0]["support_span"] == (
        "Technology improves information access, a positive effect for society."
    )
    assert support[0]["support_span"] in scores[1]["window_text"]


def test_deterministic_support_rejects_duplicate_or_incoherent_passage() -> None:
    proposal = _proposal(
        source_fold=0,
        source_document_sha256="duplicate-sha",
        obligation_id="219-positive:f0:o1:0",
    )
    scores = [
        {
            "variant": "219-positive:f0:o1:0",
            "document_id": "duplicate-copy",
            "document_sha256": "duplicate-sha",
            "fold": 1,
            "score": 9.0,
            "window_id": "duplicate",
            "window_text": "Technology has positive effects on society.",
        },
        {
            "variant": "219-positive:f0:o1:0",
            "document_id": "incoherent",
            "document_sha256": "different-sha",
            "fold": 1,
            "score": 8.0,
            "window_id": "incoherent",
            "window_text": "A product review discusses technology stock prices.",
        },
    ]

    assert qualify_opposite_support(proposal, _parent(), scores) == []


def test_discovery_preflight_rejects_existing_output_before_source_or_model(
    tmp_path: object,
) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    output = root / "discovery"
    output.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        run_discovery_preflight(
            contract_dir=root / "missing-contract",
            scores_dir=root / "missing-scores",
            output_dir=output,
        )


def test_corrected_proposal_pass_marker_forbids_any_further_retry(
    tmp_path: object,
) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    (root / "corrected_pass_started.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="one-pass"):
        run_proposal_pass(root)


def test_corrected_pass_failure_freezes_discovery_unavailable_create_only(
    tmp_path: object,
) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    (root / "corrected_pass_started.json").write_text(
        '{"expected_generation_count":48,"no_further_retry_authorized":true}',
        encoding="utf-8",
    )
    receipt = record_discovery_unavailable(
        root,
        parent_id="219-business",
        source_fold=0,
        messages_sha256="a" * 64,
        error_type="ValueError",
        error_message="local model completion is not one exact JSON value",
        cause_type="JSONDecodeError",
        cause_message="Unterminated string",
    )
    assert receipt["status"] == "discovery_unavailable"
    assert receipt["completed_proposal_pass_count"] == 0
    assert receipt["total_qwen_proposal_generation_call_count"] == 2
    assert receipt["no_further_retry_authorized"] is True
    assert verify_discovery_terminal(root)["status"] == "discovery_unavailable"
    (root / "corrected_pass_failure.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        verify_discovery_terminal(root)
    with pytest.raises(FileExistsError, match="terminal"):
        record_discovery_unavailable(
            root,
            parent_id="219-business",
            source_fold=0,
            messages_sha256="a" * 64,
            error_type="ValueError",
            error_message="failure",
            cause_type="JSONDecodeError",
            cause_message="failure",
        )


def test_finalize_records_uses_only_deterministic_opposite_fold_support() -> None:
    parent = _parent()
    proposal = _proposal(
        label="information access benefits",
        source_fold=0,
        source_document_id="source-doc",
        source_document_sha256="source-sha",
        obligation_id="219-positive:f0:o1:0",
    )
    scores = [
        {
            "variant": "219-positive:f0:o1:0",
            "document_id": "opposite",
            "document_sha256": "opposite-sha",
            "fold": 1,
            "score": 8.0,
            "window_id": "opposite-window",
            "window_text": (
                "Technology improves information access, a positive effect for society."
            ),
        }
    ]
    result = finalize_discovery_records(
        proposals=[proposal],
        nuggets=[_nugget("technology improves information access")],
        parents=[parent],
        broad_by_topic={"219": {"text": "full narrative"}},
        score_rows=scores,
    )

    assert len(result["accepted_o1"]) == 1
    accepted = result["accepted_o1"][0]
    assert accepted["validation_method"] == "deterministic_opposite_fold_minilm"
    assert accepted["opposite_fold_support"][0]["document_id"] == "opposite"
    assert accepted["query"].endswith(
        "Corpus-derived sub-obligation:\ninformation access benefits"
    )
    assert len(result["accepted_n1"]) == 1


def test_finalize_records_emits_unsupported_without_corroboration() -> None:
    result = finalize_discovery_records(
        proposals=[_proposal(source_fold=0)],
        nuggets=[],
        parents=[_parent()],
        broad_by_topic={"219": {"text": "full narrative"}},
        score_rows=[],
    )

    assert result["accepted_o1"] == []
    assert result["rejected_o1"][0]["status"] == "unsupported"
    assert "cross_fold_support" in result["rejected_o1"][0]["reasons"]
