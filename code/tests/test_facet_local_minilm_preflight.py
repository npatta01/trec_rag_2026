from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import trec_rag.facet_local_minilm_preflight as module
from trec_rag.facet_local_minilm_preflight import (
    ALLOW_PATTERNS,
    MODEL_ID,
    MODEL_REVISION,
    WindowPlanRow,
    approval_allow_patterns_sha256,
    build_benchmark_plan,
    build_preflight,
    build_window_plan,
    materialize_model,
    round_half_up_ratio,
    score_cache_context,
    verify_materialization_receipt,
    verify_window_plan,
)
from trec_rag.rerank_score_cache import GlobalScoreCache


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


class WordTokenizer:
    """Small deterministic tokenizer implementing the interface under test."""

    def __init__(self, *, pair_special_tokens: int = 3) -> None:
        self.pair_special_tokens = pair_special_tokens
        self.calls = 0

    def encode(self, text, *, add_special_tokens=False, truncation=False):
        self.calls += 1
        assert add_special_tokens is False
        assert truncation is False
        return text.split()

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        self.calls += 1
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return " ".join(token_ids)

    def num_special_tokens_to_add(self, *, pair):
        self.calls += 1
        assert pair is True
        return self.pair_special_tokens


class ExpandingDecodeTokenizer(WordTokenizer):
    """Models a tokenizer whose decoded token slice is not token-count stable."""

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        self.calls += 1
        return " ".join(f"{token} expanded" for token in token_ids)


def _candidate(
    token_count: int = 900,
    *,
    topic_id: str = "200",
    variant: str = "facet:test",
    rank: int = 1,
    document_id: str = "doc-1",
    query: str = "anchored facet",
) -> dict[str, object]:
    text = " ".join(f"t{index}" for index in range(token_count))
    return {
        "schema_version": "facet-local-minilm-candidate-row-v1",
        "topic_id": topic_id,
        "family": "facet",
        "variant": variant,
        "query": query,
        "query_sha256": _sha256(query.encode("utf-8")),
        "rank": rank,
        "document_id": document_id,
        "text": text,
        "text_sha256": _sha256(text.encode("utf-8")),
        "source_score": 1.0,
    }


def _approval() -> dict[str, object]:
    return {
        "schema_version": "facet-local-minilm-model-download-approval-v1",
        "approval_scope": "facet_local_minilm_model_materialization_v1",
        "approved_by": "user",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "allow_patterns": list(ALLOW_PATTERNS),
        "allow_patterns_sha256": approval_allow_patterns_sha256(),
        "acknowledged_network_download": True,
        "acknowledged_safe_files_only": True,
        "acknowledged_no_model_or_tokenizer_construction": True,
        "acknowledged_no_inference_qrels_retrieval_or_paid_calls": True,
    }


def _write_approval(path: Path, payload: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(payload or _approval()))


def _safe_snapshot(path: Path) -> Path:
    path.mkdir(parents=True)
    for index, name in enumerate(ALLOW_PATTERNS, start=1):
        (path / name).write_bytes(f"safe-file-{index}\n".encode("utf-8"))
    return path


def test_window_policy_is_bounded_and_covers_document_edges():
    tokenizer = WordTokenizer()
    candidate = _candidate()

    rows = build_window_plan(candidate, tokenizer, query="anchored facet")

    assert 1 <= len(rows) <= 32
    assert rows[0].document_start_token == 0
    assert rows[-1].document_end_token == rows[-1].document_token_count == 900
    assert all(row.pair_token_count <= 512 for row in rows)
    verify_window_plan(rows)


def test_overlong_query_fails_without_truncation():
    with pytest.raises(ValueError, match="query exceeds 192 tokens"):
        build_window_plan(
            _candidate(), WordTokenizer(), query=" ".join(["token"] * 193)
        )


def test_passages_overlap_by_64_tokens_and_budget_is_at_least_256():
    rows = build_window_plan(_candidate(), WordTokenizer(), query="anchored facet")

    assert rows[0].passage_token_budget == 507
    assert rows[1].document_start_token == rows[0].document_end_token - 64
    assert all(row.passage_token_budget >= 256 for row in rows)

    with pytest.raises(ValueError, match="passage budget is below 256 tokens"):
        build_window_plan(
            _candidate(),
            WordTokenizer(pair_special_tokens=65),
            query=" ".join(["query"] * 192),
        )


def test_window_fitting_uses_retokenized_serialized_passage_length():
    tokenizer = ExpandingDecodeTokenizer()
    rows = build_window_plan(_candidate(900), tokenizer, query="anchored facet")

    assert rows[1].document_start_token == rows[0].document_end_token - 64
    for row in rows:
        actual_passage_tokens = tokenizer.encode(
            row.window_text, add_special_tokens=False, truncation=False
        )
        actual_pair_count = (
            row.query_token_count
            + len(actual_passage_tokens)
            + row.pair_special_token_count
        )
        assert row.pair_token_count == actual_pair_count
        assert actual_pair_count <= 512


def test_capped_windows_use_exact_half_up_rule_and_unique_indices():
    tokenizer = WordTokenizer()
    candidate = _candidate(44_364)

    rows = build_window_plan(candidate, tokenizer, query="anchored facet")
    original_count = rows[0].original_window_count
    expected = tuple(
        round_half_up_ratio(index * (original_count - 1), 31)
        for index in range(32)
    )

    assert original_count > 32
    assert tuple(row.original_window_index for row in rows) == expected
    assert len(expected) == len(set(expected)) == 32
    assert expected[0] == 0
    assert expected[-1] == original_count - 1


def test_window_ids_hashes_and_cache_keys_are_stable_and_content_bound(tmp_path):
    candidate = _candidate()
    rows = build_window_plan(candidate, WordTokenizer(), query="anchored facet")
    repeated = build_window_plan(candidate, WordTokenizer(), query="anchored facet")
    cache = GlobalScoreCache(tmp_path, score_cache_context())

    assert rows == repeated
    assert len({row.window_id for row in rows}) == len(rows)
    for row in rows:
        assert row.query_sha256 == _sha256(row.query.encode("utf-8"))
        assert row.document_sha256 == _sha256(candidate["text"].encode("utf-8"))
        assert row.window_sha256 == _sha256(row.window_text.encode("utf-8"))
        assert row.cache_key == cache.cache_key(
            query_text=row.query, text=row.window_text
        )

    changed = build_window_plan(
        {**candidate, "document_id": "doc-other"},
        WordTokenizer(),
        query="anchored facet",
    )
    assert [row.window_id for row in rows] != [row.window_id for row in changed]
    assert [row.cache_key for row in rows] == [row.cache_key for row in changed]


def test_materialization_requires_exact_canonical_approval_before_download(tmp_path):
    approval_path = tmp_path / "approval.json"
    output = tmp_path / "model-v1"
    download_calls = []

    def forbidden_download(**kwargs):
        download_calls.append(kwargs)
        pytest.fail("download must stay behind approval")

    with pytest.raises(ValueError, match="model download approval"):
        materialize_model(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            approval_path=approval_path,
            output_dir=output,
            snapshot_download_fn=forbidden_download,
        )
    assert download_calls == []

    wrong = _approval()
    wrong["revision"] = "0" * 40
    _write_approval(approval_path, wrong)
    with pytest.raises(ValueError, match="revision mismatch"):
        materialize_model(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            approval_path=approval_path,
            output_dir=output,
            snapshot_download_fn=forbidden_download,
        )
    assert download_calls == []

    approval_path.write_text(json.dumps(_approval()), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON bytes"):
        materialize_model(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            approval_path=approval_path,
            output_dir=output,
            snapshot_download_fn=forbidden_download,
        )
    assert download_calls == []


def test_materialization_downloads_only_allowlist_without_constructing_model(tmp_path):
    approval_path = tmp_path / "approval.json"
    output = tmp_path / "model-v1"
    snapshot = _safe_snapshot(tmp_path / "snapshot")
    _write_approval(approval_path)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    receipt = materialize_model(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        approval_path=approval_path,
        output_dir=output,
        snapshot_download_fn=snapshot_download,
    )

    assert calls == [
        {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "allow_patterns": list(ALLOW_PATTERNS),
        }
    ]
    assert [row["path"] for row in receipt["files"]] == list(ALLOW_PATTERNS)
    assert receipt["snapshot_bytes"] == sum(
        (snapshot / name).stat().st_size for name in ALLOW_PATTERNS
    )
    assert verify_materialization_receipt(output / "materialization.json") == snapshot

    with pytest.raises(FileExistsError, match="create-only"):
        materialize_model(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            approval_path=approval_path,
            output_dir=output,
            snapshot_download_fn=lambda **_kwargs: pytest.fail("download on replay"),
        )


def test_materialization_rejects_pickle_and_unexpected_files(tmp_path):
    approval_path = tmp_path / "approval.json"
    _write_approval(approval_path)
    snapshot = _safe_snapshot(tmp_path / "snapshot")
    (snapshot / "pytorch_model.bin").write_bytes(b"pickle-like")

    with pytest.raises(ValueError, match="pickle or unexpected model files"):
        materialize_model(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            approval_path=approval_path,
            output_dir=tmp_path / "model-v1",
            snapshot_download_fn=lambda **_kwargs: str(snapshot),
        )


def test_verified_tokenizer_load_is_local_only_and_receipt_first(tmp_path):
    approval_path = tmp_path / "approval.json"
    output = tmp_path / "model-v1"
    snapshot = _safe_snapshot(tmp_path / "snapshot")
    _write_approval(approval_path)
    materialize_model(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        approval_path=approval_path,
        output_dir=output,
        snapshot_download_fn=lambda **_kwargs: str(snapshot),
    )
    calls = []

    class AutoTokenizerSpy:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((Path(path), kwargs))
            return "tokenizer"

    assert (
        module.load_verified_tokenizer(
            output / "materialization.json", auto_tokenizer_cls=AutoTokenizerSpy
        )
        == "tokenizer"
    )
    assert calls == [
        (
            snapshot,
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_fast": True,
            },
        )
    ]

    original_size = (snapshot / "tokenizer.json").stat().st_size
    (snapshot / "tokenizer.json").write_bytes(b"x" * original_size)
    calls.clear()
    with pytest.raises(ValueError, match="materialized file hash mismatch"):
        module.load_verified_tokenizer(
            output / "materialization.json", auto_tokenizer_cls=AutoTokenizerSpy
        )
    assert calls == []


class _EmptyCache:
    def __init__(self) -> None:
        self.scores = {}
        self.calls = 0

    def cache_key(self, *, query_text, text):
        self.calls += 1
        return module._score_cache_key(query_text, text)


def test_protected_topic_rejected_before_tokenizer_or_cache_access():
    tokenizer = WordTokenizer()
    cache = _EmptyCache()

    with pytest.raises(ValueError, match="protected topic 144"):
        build_preflight([_candidate(topic_id="144")], tokenizer, cache)

    assert tokenizer.calls == 0
    assert cache.calls == 0


def test_preflight_reports_caps_coverage_stream_quantiles_and_ceiling():
    tokenizer = WordTokenizer()
    candidates = [
        _candidate(100, rank=1, document_id="short"),
        _candidate(700, rank=2, document_id="medium"),
        _candidate(14_683, rank=3, document_id="capped"),
    ]
    plan = build_preflight(candidates, tokenizer, _EmptyCache())

    stats = plan.streams[0]
    fractions = [row.document_token_coverage_fraction for row in plan.documents]
    assert stats["window_count"] == len(plan.windows)
    assert stats["capped_document_count"] == 1
    assert stats["coverage_min"] == min(fractions)
    assert stats["coverage_median"] == fractions[1]
    assert stats["coverage_p95"] == max(fractions)
    assert plan.summary["capped_document_count"] == 1
    assert plan.summary["cache_miss_count"] == len(plan.windows)
    assert plan.summary["unique_uncached_pair_count"] == len(
        {row.cache_key for row in plan.windows}
    )

    with pytest.raises(ValueError, match="100,000-miss ceiling"):
        module.enforce_uncached_pair_ceiling(100_001)


def test_benchmark_sample_is_hash_frozen_and_deterministic_under_reordering():
    candidates = []
    for index in range(100):
        candidate = _candidate(
            1 + (index % 20),
            rank=index + 1,
            document_id=f"doc-{index:03d}",
        )
        candidate["text"] = f"unique-{index} {candidate['text']}"
        candidate["text_sha256"] = _sha256(candidate["text"].encode("utf-8"))
        candidates.append(candidate)
    first = build_preflight(candidates, WordTokenizer(), _EmptyCache())
    second = build_preflight(list(reversed(candidates)), WordTokenizer(), _EmptyCache())

    assert [row.to_dict() for row in first.windows] == [
        row.to_dict() for row in second.windows
    ]
    assert first.summary == second.summary
    assert first.benchmark == second.benchmark
    assert first.benchmark["warmup_pair_count"] == 32
    assert first.benchmark["timed_sample_pair_count"] == 64
    assert first.benchmark["timed_repetitions"] == 3
    assert first.benchmark["forward_pair_count"] == 224
    assert first.benchmark["sample_sha256"] == _sha256(
        module.canonical_compact_json_bytes(
            {
                "warmup_cache_keys": first.benchmark["warmup_cache_keys"],
                "timed_cache_keys": first.benchmark["timed_cache_keys"],
                "timed_repetitions": 3,
            }
        )
    )


def test_small_and_zero_miss_benchmark_rules_are_deterministic():
    template = build_window_plan(_candidate(1), WordTokenizer(), query="facet")[0]
    rows = tuple(
        replace(
            template,
            window_id=f"{index:064x}",
            cache_key=f"{index:064x}",
            pair_token_count=index + 1,
            cache_hit=False,
        )
        for index in range(20)
    )

    small = build_benchmark_plan(rows)
    assert small["mode"] == "small_sample"
    assert small["warmup_pair_count"] == 5
    assert small["timed_sample_pair_count"] == 15
    assert small["timed_repetitions"] == 1
    assert small["forward_pair_count"] == 20

    zero = build_benchmark_plan(tuple(replace(row, cache_hit=True) for row in rows))
    assert zero["mode"] == "cache_complete"
    assert zero["forward_pair_count"] == 0


def test_window_row_schema_round_trips_and_verifier_detects_drift():
    row = build_window_plan(_candidate(10), WordTokenizer(), query="facet")[0]
    loaded = WindowPlanRow.from_dict(row.to_dict())
    assert loaded == row
    verify_window_plan((loaded,))

    with pytest.raises(ValueError, match="window hash mismatch"):
        verify_window_plan((replace(row, window_sha256="0" * 64),))


def test_cli_has_no_qrels_retrieval_or_inference_input():
    parser = module.build_argument_parser()
    destinations = {action.dest for action in parser._actions}

    assert "qrels" not in destinations
    assert "retrieval" not in destinations
    assert "model" not in destinations
