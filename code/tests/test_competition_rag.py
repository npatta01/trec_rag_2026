from __future__ import annotations

import asyncio
import copy
import json
import os
import stat
import threading
import time
import zipfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
import httpx_retries.retry as httpx_retry_module
import pytest

import trec_rag.competition_rag as competition_rag
from trec_rag.competition_rag import (
    OpenRouterJsonGenerator,
    RagGenerationConfig,
    arguments,
    build_submission_record,
    load_rag_generation_config,
    parse_generated_json,
    run_generation,
    validate_submission_record,
)
from trec_rag.facet_pilot_config import load_facet_pilot_config
from trec_rag.generation_handoff import (
    SOURCE_CONTRACT,
    ClaimHint,
    EvidenceGroup,
    EvidencePassage,
    EvidenceSourceSpan,
    GenerationHandoff,
    GenerationTopic,
    HandoffProducer,
    SelectedCluster,
    TopicSourceReceipts,
    load_generation_handoff,
    select_generation_topics,
    serialize_generation_handoff,
    write_generation_handoff,
)

def _config_text(
    *,
    experiment_extra: str = "",
    inputs_extra: str = "",
    generation_extra: str = "",
    root_extra: str = "",
) -> str:
    return f"""schema_version: competition_rag_config_v2
experiment:
  id: rag26_competition_rag_gpt_sol_v2
  output_dir: outputs/rag26_competition_rag_gpt_sol_v2
  mode: create
{experiment_extra}submission:
  team_id: castorini
  run_desc: Fixed retrieval with GPT-5.6 Sol answer generation.
inputs:
  handoff_manifest: outputs/facet-deepseek-b40-v2/generation_handoff_manifest.json
{inputs_extra}generation:
  type: openrouter
  api_base: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  model: openai/gpt-5.6-sol
  reasoning_effort: medium
  temperature: null
  max_tokens: 6000
  timeout_seconds: 900
  transport_max_attempts: 3
  concurrency: 4
{generation_extra}
{root_extra}"""


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "competition-rag.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_strict_generation_config_and_resolves_inputs_from_checkout(
    tmp_path: Path,
) -> None:
    config = load_rag_generation_config(_write_config(tmp_path, _config_text()))

    assert config.schema_version == "competition_rag_config_v2"
    assert config.handoff_manifest_path == (
        tmp_path
        / "outputs/facet-deepseek-b40-v2/generation_handoff_manifest.json"
    )
    assert config.topic_ids is None
    assert config.strategy == "baseline"
    assert config.output_path == (
        tmp_path / "outputs/rag26_competition_rag_gpt_sol_v2/rag_output_trec_rag_2026.jsonl"
    )
    assert config.run_id == "rag26_competition_rag_gpt_sol_v2"


@pytest.mark.parametrize(
    "output_dir",
    [
        "/tmp/competition-rag-output",
        "../competition-rag-output",
        "outputs/../competition-rag-output",
        "outputs",
    ],
)
def test_config_rejects_output_directories_outside_a_dedicated_outputs_child(
    tmp_path: Path, output_dir: str
) -> None:
    content = _config_text().replace(
        "  output_dir: outputs/rag26_competition_rag_gpt_sol_v2\n",
        f"  output_dir: {output_dir}\n",
    )

    with pytest.raises(ValueError, match="experiment.output_dir"):
        load_rag_generation_config(_write_config(tmp_path, content))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (_config_text().replace("schema_version: competition_rag_config_v2\n", ""), "schema_version"),
        (_config_text().replace("competition_rag_config_v2", "other_schema"), "schema_version"),
        (_config_text(root_extra="unexpected: true\n"), "unknown"),
        (_config_text().replace("submission:\n  team_id: castorini\n  run_desc: Fixed retrieval with GPT-5.6 Sol answer generation.\n", ""), "missing"),
        (_config_text().replace("  concurrency: 4\n", "  concurrency: 4\n  concurrency: 2\n"), "duplicate YAML key"),
    ],
)
def test_config_rejects_missing_unknown_wrong_schema_and_duplicate_keys(
    tmp_path: Path, content: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_rag_generation_config(_write_config(tmp_path, content))


@pytest.mark.parametrize(
    "old_field",
    [
        "  queries: official/topics.tsv\n",
        "  run: outputs/retrieval/run.tsv\n",
        "  documents: outputs/retrieval/documents.zip\n",
        "  archive_member: documents.jsonl\n",
    ],
)
def test_config_rejects_every_old_generation_input_field(
    tmp_path: Path, old_field: str
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        load_rag_generation_config(
            _write_config(tmp_path, _config_text(inputs_extra=old_field))
        )


def test_config_rejects_old_retrieval_and_prompt_profile_sections(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        load_rag_generation_config(
            _write_config(
                tmp_path,
                _config_text(root_extra="retrieval:\n  top_k: 100\n"),
            )
        )
    with pytest.raises(ValueError, match="unknown"):
        load_rag_generation_config(
            _write_config(
                tmp_path,
                _config_text(generation_extra="  prompt_profile: focused_citations_tail\n"),
        )
    )


def test_config_loads_coverage_aware_generation_strategy(tmp_path: Path) -> None:
    config = load_rag_generation_config(
        _write_config(
            tmp_path,
            _config_text(generation_extra="  strategy: coverage_aware\n"),
        )
    )

    assert config.strategy == "coverage_aware"


def test_config_rejects_unknown_generation_strategy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="generation.strategy is unsupported"):
        load_rag_generation_config(
            _write_config(
                tmp_path,
                _config_text(generation_extra="  strategy: open_ended_agent\n"),
            )
        )


def test_config_places_topic_subset_under_experiment(tmp_path: Path) -> None:
    config = load_rag_generation_config(
        _write_config(
            tmp_path,
            _config_text(experiment_extra="  topic_ids: [rag2026-58, rag2026-200]\n"),
        )
    )

    assert config.topic_ids == ("rag2026-58", "rag2026-200")


@pytest.mark.parametrize(
    "filename",
    [
        "rag26_competition_rag_gpt_sol_v2.yaml",
        "rag26_competition_rag_deepseek_v2.yaml",
    ],
)
def test_checked_in_competition_configs_use_only_the_full_handoff(
    filename: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_rag_generation_config(repo_root / "configs" / filename)
    retrieval = load_facet_pilot_config(
        repo_root / "configs/rag26_competition_retrieval_v2.yaml"
    )

    assert config.topic_ids is None
    assert config.strategy == "baseline"
    assert config.handoff_manifest_path == (
        retrieval.output_dir / "generation_handoff_manifest.json"
    )
    assert not hasattr(config, "queries_path")
    assert not hasattr(config, "run_path")
    assert not hasattr(config, "documents_path")


def _generated_record() -> dict[str, Any]:
    return {
        "references": ["climbmix-a", "climbmix-b"],
        "answer": [
            {"text": "The first source supports the finding.", "citations": [0]},
            {"text": "The second source adds a limitation.", "citations": [1]},
        ],
    }


def test_provider_schema_uses_supported_strict_cardinality_constraints() -> None:
    schema = competition_rag.output_schema()
    references = schema["properties"]["references"]
    answer = schema["properties"]["answer"]
    citations = answer["items"]["properties"]["citations"]

    assert references["minItems"] == 1
    assert answer["minItems"] == 1
    assert citations["minItems"] == 1
    assert citations["maxItems"] == 3
    assert "uniqueItems" not in json.dumps(schema)


def test_v2_provider_schema_and_prompt_require_raw_docids() -> None:
    schema = competition_rag.output_schema()
    citation_schema = schema["properties"]["answer"]["items"]["properties"]["citations"]
    assert citation_schema["items"] == {"type": "string", "minLength": 1}
    topic = _generation_topic(
        "rag2026-0",
        "What does the evidence show?",
        [("climbmix-a", "Selected evidence.")],
    )
    prompt = competition_rag.render_prompt(topic)
    assert "raw ClimbMix docid" in prompt
    assert "do not use numeric citation indexes" in prompt.lower()


def test_coverage_aware_prompt_adds_handoff_only_checklist_and_one_audit() -> None:
    topic = _generation_topic(
        "rag2026-0",
        "What does the evidence show?",
        [
            ("climbmix-a", "First selected passage."),
            ("climbmix-b", "Second selected passage."),
        ],
    )

    baseline = competition_rag.render_prompt(topic)
    coverage = competition_rag.render_prompt(topic, strategy="coverage_aware")

    assert "Ordered answer checklist:" not in baseline
    assert "Ordered answer checklist:" in coverage
    assert "[rag2026-0-g1] Selected evidence for rag2026-0." in coverage
    assert "selected passages: 2; advisory claim hints: 1" in coverage
    assert "perform exactly one private audit" in coverage
    assert "roughly 900 to 1,000 answer words" in coverage
    assert "First selected passage." in coverage
    assert "Second selected passage." in coverage
    assert "gold nugget" not in coverage.lower()
    assert "qrels" not in coverage.lower()


def _submission_record() -> dict[str, Any]:
    return {
        "metadata": {
            "team_id": "castorini",
            "narrative_id": "rag2026-0",
            "narrative": "Official question",
            "run_id": "rag26_competition_rag_gpt_sol_v1",
            "run_desc": "Fixed retrieval with GPT-5.6 Sol answer generation.",
        },
        **_generated_record(),
    }


def _validate_submission(record: dict[str, Any]) -> None:
    validate_submission_record(
        record,
        topic_id="rag2026-0",
        narrative="Official question",
        allowed_docids=["climbmix-a", "climbmix-b", "climbmix-c"],
        team_id="castorini",
        run_id="rag26_competition_rag_gpt_sol_v1",
        run_desc="Fixed retrieval with GPT-5.6 Sol answer generation.",
    )


def test_validates_organizer_shaped_submission_record() -> None:
    _validate_submission(_submission_record())


def test_validates_all_organizer_allowed_rag_citation_forms_and_metadata() -> None:
    record = _submission_record()
    record["metadata"]["participant_note"] = "extra metadata is organizer-valid"
    record["references"] = ["climbmix-a", "climbmix-b", "climbmix-c"]
    record["answer"] = [
        {"text": "This heading is intentionally uncited.", "citations": []},
        {"text": "This claim uses a reference index.", "citations": [0]},
        {"text": "This claim uses a direct document ID.", "citations": ["climbmix-b"]},
    ]

    _validate_submission(record)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update({"extra": True}), "root object or metadata"),
        (lambda row: row["references"].append("climbmix-a"), "duplicated or outside"),
        (lambda row: row["references"].append("not-selected"), "duplicated or outside"),
        (lambda row: row["answer"][0].update({"citations": [True]}), "citation"),
        (lambda row: row["answer"][0].update({"citations": [2]}), "citation"),
        (lambda row: row["answer"][0].update({"citations": ["not-selected"]}), "citation"),
        (lambda row: row["answer"][0].update({"citations": [0, 1, 0, 1]}), "0-3"),
        (lambda row: row["answer"][0].update({"heading": "Finding"}), "invalid fields"),
    ],
)
def test_rejects_submission_records_outside_the_organizer_contract(
    mutate: Any, message: str
) -> None:
    record = copy.deepcopy(_submission_record())
    mutate(record)

    with pytest.raises(ValueError, match=message):
        _validate_submission(record)


def test_rejects_answer_over_1024_whitespace_words() -> None:
    record = _submission_record()
    record["references"] = ["climbmix-a"]
    record["answer"] = [{"text": "word " * 1025, "citations": [0]}]

    with pytest.raises(ValueError, match="1,024 words"):
        _validate_submission(record)


@pytest.mark.parametrize("extra_key", ["metadata", "model_note", "unexpected"])
def test_generated_root_must_contain_exactly_references_and_answer(
    extra_key: str,
) -> None:
    generated = _generated_record()
    generated[extra_key] = {"ignored": "must be rejected before metadata injection"}

    with pytest.raises(ValueError, match="generated root"):
        build_submission_record(
            generated,
            topic_id="rag2026-0",
            narrative="Official question",
            team_id="castorini",
            run_id="rag26_competition_rag_gpt_sol_v1",
            run_desc="Fixed retrieval with GPT-5.6 Sol answer generation.",
        )


def test_build_submission_record_injects_exact_official_metadata() -> None:
    record = build_submission_record(
        _generated_record(),
        topic_id="rag2026-0",
        narrative="Official question",
        team_id="castorini",
        run_id="rag26_competition_rag_gpt_sol_v1",
        run_desc="Fixed retrieval with GPT-5.6 Sol answer generation.",
    )

    assert list(record) == ["metadata", "references", "answer"]
    assert list(record["metadata"]) == [
        "team_id",
        "narrative_id",
        "narrative",
        "run_id",
        "run_desc",
    ]


def test_parses_only_one_plain_or_fenced_json_object() -> None:
    assert parse_generated_json('{"references": [], "answer": []}') == {
        "references": [],
        "answer": [],
    }
    assert parse_generated_json('```json\n{"references": [], "answer": []}\n```') == {
        "references": [],
        "answer": [],
    }
    with pytest.raises(ValueError, match="one JSON object"):
        parse_generated_json('before {"answer": []} after')


class FakeGenerator:
    def __init__(self, outputs: dict[str, dict[str, Any]]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self,
        *,
        topic_id: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append(
            {
                "topic_id": topic_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_schema": response_schema,
            }
        )
        return copy.deepcopy(self.outputs[topic_id]), {"id": f"fake-{topic_id}"}


def _generation_topic(
    topic_id: str,
    narrative: str,
    documents: list[tuple[str, str]],
) -> GenerationTopic:
    evidence: list[EvidencePassage] = []
    clusters: list[SelectedCluster] = []
    for ordinal, (docid, text) in enumerate(documents, start=1):
        evidence_id = f"{topic_id}-e{ordinal}"
        cluster_id = f"{topic_id}-c{ordinal}"
        start = ordinal * 100
        evidence.append(
            EvidencePassage(
                evidence_id=evidence_id,
                group_id=f"{topic_id}-g1",
                cluster_id=cluster_id,
                cluster_ordinal=ordinal,
                support_ordinal=1,
                candidate_kind="extractive",
                docid=docid,
                document_rank=ordinal,
                text=text,
                document_sha256=sha256(f"{topic_id}:{docid}".encode()).hexdigest(),
                source_span=EvidenceSourceSpan(
                    start_char=start,
                    end_char=start + len(text),
                    start_byte=start,
                    end_byte=start + len(text.encode("utf-8")),
                ),
            )
        )
        clusters.append(
            SelectedCluster(
                cluster_id=cluster_id,
                ordinal=ordinal,
                representative_evidence_id=evidence_id,
                evidence_ids=(evidence_id,),
            )
        )
    return GenerationTopic(
        topic_id=topic_id,
        narrative=narrative,
        groups=(
            EvidenceGroup(
                group_id=f"{topic_id}-g1",
                kind="generated_subnarrative",
                text=f"Selected evidence for {topic_id}.",
                selected_clusters=tuple(clusters),
            ),
        ),
        evidence=tuple(evidence),
        claim_hints=(
            ClaimHint(
                claim_id=f"{topic_id}-claim-1",
                group_id=f"{topic_id}-g1",
                kind="canonical",
                text=f"Evidence was selected for {topic_id}.",
                evidence_ids=(evidence[0].evidence_id,),
            ),
        ),
        source_receipts=TopicSourceReceipts(
            official_topics_sha256="1" * 64,
            retrieval_topic_sha256=sha256(topic_id.encode()).hexdigest(),
        ),
    )


def _write_handoff(tmp_path: Path, topics: tuple[GenerationTopic, ...]) -> Path:
    path = (
        tmp_path
        / "outputs/facet-deepseek-b40-v2/generation_handoff_manifest.json"
    )
    write_generation_handoff(
        path,
        GenerationHandoff(
            producer=HandoffProducer(
                source_contract=SOURCE_CONTRACT,
                retrieval_run_id="facet-deepseek-b40-v2",
                producer_revision="test-revision",
            ),
            topics=topics,
        ),
    )
    return path


def _pipeline_config(tmp_path: Path) -> RagGenerationConfig:
    _write_handoff(
        tmp_path,
        (
            _generation_topic("rag2026-2", "Question two", [("climbmix-c", "Evidence C.")]),
            _generation_topic(
                "rag2026-1",
                "Question one",
                [("climbmix-a", "Evidence A."), ("climbmix-b", "Evidence B.")],
            ),
        ),
    )
    return load_rag_generation_config(_write_config(tmp_path, _config_text()))


def _topic_output(docids: list[str], text: str) -> dict[str, Any]:
    return {
        "references": docids,
        "answer": [{"text": text, "citations": list(docids)}],
    }


def test_main_authenticates_the_generation_handoff_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    config_path = tmp_path / "canonical-config.yaml"
    load_calls = 0
    real_load = competition_rag.load_generation_handoff

    def counted_load(path: Path) -> GenerationHandoff:
        nonlocal load_calls
        load_calls += 1
        return real_load(path)

    generator = FakeGenerator(
        {"rag2026-1": _topic_output(["climbmix-a"], "Evidence A supports it.")}
    )
    monkeypatch.setattr(competition_rag, "arguments", lambda: config_path)
    monkeypatch.setattr(
        competition_rag,
        "load_rag_generation_config",
        lambda _path: config,
    )
    monkeypatch.setattr(competition_rag, "load_generation_handoff", counted_load)
    monkeypatch.setattr(competition_rag, "find_repo_root", lambda _path: tmp_path)
    monkeypatch.setattr(competition_rag, "load_repo_env", lambda _root: None)
    monkeypatch.setattr(
        competition_rag,
        "OpenRouterJsonGenerator",
        lambda **_kwargs: generator,
    )

    competition_rag.main()

    assert load_calls == 1


def test_prompt_uses_only_manifest_narrative_and_selected_evidence() -> None:
    topic = _generation_topic(
        "rag2026-58",
        "What does the selected evidence show?",
        [("climbmix-a", "Selected passage A."), ("climbmix-b", "Selected passage B.")],
    )
    full_document_not_selected = "FULL_DOCUMENT_SENTINEL"

    prompt = competition_rag.render_prompt(topic)

    assert topic.narrative in prompt
    assert "Selected passage A." in prompt
    assert "Selected passage B." in prompt
    assert "docid=climbmix-a" in prompt
    assert "docid=climbmix-b" in prompt
    assert full_document_not_selected not in prompt
    flat = " ".join(prompt.split())
    assert "one self-contained sentence" in flat
    assert "one to three unique raw ClimbMix docid" in flat
    assert "never exceed 1,024" in flat


def test_normalizer_rebuilds_references_from_citations_without_touching_text() -> None:
    record = {
        "metadata": {"narrative_id": "58"},
        # b and d are never cited; a and c are, out of order.
        "references": ["doc-a", "doc-b", "doc-c", "doc-d"],
        "answer": [
            {"text": "Second reference first.", "citations": [2]},
            {"text": "Then the first one.", "citations": [0, 2]},
        ],
    }

    out = competition_rag.normalize_generated_record(record)

    # References are exactly the cited docids, ordered by first use.
    assert out["references"] == ["doc-c", "doc-a"]
    # Every reference is now cited, which is what the strict profile demands.
    used = {c for item in out["answer"] for c in item["citations"]}
    assert used == set(range(len(out["references"])))
    # Citations still point at the same documents they did before.
    assert [out["references"][c] for c in out["answer"][0]["citations"]] == ["doc-c"]
    assert [out["references"][c] for c in out["answer"][1]["citations"]] == ["doc-a", "doc-c"]
    # Answer text is untouched.
    assert [i["text"] for i in out["answer"]] == [i["text"] for i in record["answer"]]


def test_raw_docid_normalizer_maps_first_use_and_rejects_numeric_or_foreign() -> None:
    record = {
        "metadata": {},
        "references": ["doc-b"],
        "answer": [
            {"text": "First claim.", "citations": ["doc-a"]},
            {"text": "Second claim.", "citations": ["doc-b", "doc-a", "doc-a"]},
        ],
    }
    out = competition_rag.normalize_generated_record(
        record, allowed_docids=["doc-a", "doc-b", "doc-c"]
    )
    assert out["references"] == ["doc-a", "doc-b"]
    assert out["answer"][0]["citations"] == [0]
    assert out["answer"][1]["citations"] == [1, 0]

    numeric = {
        "references": ["doc-a"],
        "answer": [{"text": "Claim.", "citations": [0]}],
    }
    with pytest.raises(ValueError, match="raw docid strings"):
        competition_rag.normalize_generated_record(numeric, allowed_docids=["doc-a"])

    foreign = {
        "references": ["foreign"],
        "answer": [{"text": "Claim.", "citations": ["foreign"]}],
    }
    with pytest.raises(ValueError, match="outside supplied ranked documents"):
        competition_rag.normalize_generated_record(foreign, allowed_docids=["doc-a"])


def test_normalized_record_satisfies_the_strict_profile(tmp_path: Path) -> None:
    generated = {
        "references": ["climbmix-a", "climbmix-b", "climbmix-c"],
        "answer": [{"text": "A grounded claim.", "citations": [1]}],
    }
    record = competition_rag.normalize_generated_record(
        competition_rag.build_submission_record(
            generated,
            topic_id="58",
            narrative="Official question",
            team_id="local-baseline",
            run_id="dev-spike",
            run_desc="development spike",
        )
    )

    # Would have raised "uncited references" without normalization.
    competition_rag._validate_generated_submission_record(
        record,
        topic_id="58",
        narrative="Official question",
        allowed_docids=["climbmix-a", "climbmix-b", "climbmix-c"],
        team_id="local-baseline",
        run_id="dev-spike",
        run_desc="development spike",
    )
    assert record["references"] == ["climbmix-b"]


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        pytest.param(
            {"references": ["a"], "answer": ["oops"]},
            "not an object",
            id="answer-item-is-a-string",
        ),
        pytest.param(
            {"references": ["a"], "answer": [{"citations": [0]}]},
            "text",
            id="missing-text",
        ),
        pytest.param(
            {"references": ["a"], "answer": [{"text": "x", "citations": 0}]},
            "citations list",
            id="citations-not-a-list",
        ),
        pytest.param(
            {"references": "a", "answer": [{"text": "x", "citations": [0]}]},
            "references must be a nonempty list",
            id="references-not-a-list",
        ),
    ],
)
def test_post_processors_reject_malformed_shapes_cleanly(
    record: dict[str, Any], expected: str
) -> None:
    """These run before validation, so a malformed model response must not raise TypeError.

    Relaxing structured output to json_object produced exactly these shapes from a live model.
    """
    with pytest.raises(ValueError, match=expected):
        competition_rag.normalize_generated_record(
            competition_rag.trim_to_word_limit(record)
        )


def test_normalizer_collapses_a_duplicate_citation() -> None:
    record = {
        "metadata": {},
        "references": ["doc-a", "doc-b"],
        "answer": [{"text": "Claim.", "citations": [1, 1]}],
    }
    out = competition_rag.normalize_generated_record(record)
    # Would otherwise survive as [0, 0] and fail the unique-citation rule.
    assert out["answer"][0]["citations"] == [0]
    assert out["references"] == ["doc-b"]


def test_normalizer_canonicalizes_a_docid_listed_twice() -> None:
    record = {
        "metadata": {},
        "references": ["doc-a", "doc-b", "doc-a"],
        "answer": [{"text": "Claim.", "citations": [0, 2]}],
    }
    out = competition_rag.normalize_generated_record(record)
    # Both positions name the same document, so one citation of one reference remains.
    assert out["references"] == ["doc-a"]
    assert out["answer"][0]["citations"] == [0]


def test_normalizer_reports_an_invalid_citation_rather_than_dropping_it() -> None:
    """A fabricated citation index must fail loudly, not be silently pruned.

    Two of the generators tested for this track emitted document ids outside the retrieval
    pool, so this is a live failure mode rather than a hypothetical one.
    """
    record = {
        "metadata": {"narrative_id": "58"},
        "references": ["doc-a", "doc-b"],
        "answer": [
            {"text": "Grounded claim.", "citations": [0]},
            {"text": "Fabricated pointer.", "citations": [9]},
        ],
    }

    with pytest.raises(ValueError, match="answer\\[1\\] has an invalid citation"):
        competition_rag.normalize_generated_record(record)


def test_normalizer_rejects_a_partially_invalid_citation_list() -> None:
    record = {
        "metadata": {},
        "references": ["doc-a", "doc-b"],
        "answer": [{"text": "Claim.", "citations": [0, 9]}],
    }

    with pytest.raises(ValueError, match="invalid citation"):
        competition_rag.normalize_generated_record(record)


def test_word_cap_trim_drops_trailing_objects_and_frees_their_references() -> None:
    record = {
        "metadata": {"narrative_id": "58"},
        "references": ["doc-a", "doc-b"],
        "answer": [
            {"text": "word " * 900, "citations": [0]},
            {"text": "word " * 200, "citations": [1]},
        ],
    }

    trimmed = competition_rag.trim_to_word_limit(record)
    assert len(trimmed["answer"]) == 1
    assert sum(len(i["text"].split()) for i in trimmed["answer"]) <= 1024

    # The reference only the dropped object cited is normalized away.
    out = competition_rag.normalize_generated_record(trimmed)
    assert out["references"] == ["doc-a"]


def test_word_cap_trim_is_a_no_op_when_already_within_budget() -> None:
    record = {
        "metadata": {},
        "references": ["doc-a"],
        "answer": [{"text": "Short claim.", "citations": [0]}],
    }
    assert competition_rag.trim_to_word_limit(record) is record


def test_configuration_mode_guidance_uses_config_values_not_cli_flags(tmp_path: Path) -> None:
    existing = _pipeline_config(tmp_path)
    existing.output_path.parent.mkdir(parents=True, exist_ok=True)
    existing.output_path.write_text("old output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="experiment\\.mode: resume") as create_error:
        asyncio.run(run_generation(existing, FakeGenerator({})))
    assert "experiment.mode: overwrite" in str(create_error.value)
    assert "--resume" not in str(create_error.value)
    assert "--overwrite" not in str(create_error.value)

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed = _pipeline_config(failed_root)
    with pytest.raises(RuntimeError, match="experiment\\.mode: resume") as failure_error:
        asyncio.run(run_generation(failed, FakeGenerator({})))
    assert "--resume" not in str(failure_error.value)


def test_runs_generation_in_official_query_order_and_atomically_consolidates(
    tmp_path: Path,
) -> None:
    config = _pipeline_config(tmp_path)
    generator = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a", "climbmix-b"], "A and B support it."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )

    asyncio.run(run_generation(config, generator))

    rows = [json.loads(line) for line in config.output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["metadata"]["narrative_id"] for row in rows] == ["rag2026-2", "rag2026-1"]
    assert rows[0]["answer"][0]["citations"] == [0]
    assert {call["topic_id"] for call in generator.calls} == {"rag2026-1", "rag2026-2"}
    topic_one_prompt = next(call["user_prompt"] for call in generator.calls if call["topic_id"] == "rag2026-1")
    assert "docid=climbmix-a" in topic_one_prompt
    assert "docid=climbmix-b" in topic_one_prompt


def test_coverage_aware_generation_is_one_hosted_completion_per_topic(
    tmp_path: Path,
) -> None:
    config = replace(
        _pipeline_config(tmp_path),
        topic_ids=("rag2026-1",),
        strategy="coverage_aware",
    )
    generator = FakeGenerator(
        {
            "rag2026-1": _topic_output(
                ["climbmix-a"],
                "The selected evidence supports the answer.",
            )
        }
    )

    asyncio.run(run_generation(config, generator))

    assert len(generator.calls) == 1
    assert "Ordered answer checklist:" in generator.calls[0]["user_prompt"]
    assert "perform exactly one private audit" in generator.calls[0]["user_prompt"]


def test_generation_trims_an_over_limit_completion_before_validation(
    tmp_path: Path,
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generator = FakeGenerator(
        {
            "rag2026-1": {
                "references": ["climbmix-a", "climbmix-b"],
                "answer": [
                    {"text": "supported " * 900, "citations": ["climbmix-a"]},
                    {"text": "trailing " * 200, "citations": ["climbmix-b"]},
                ],
            }
        }
    )

    asyncio.run(run_generation(config, generator))

    record = json.loads(config.output_path.read_text(encoding="utf-8"))
    assert len(generator.calls) == 1
    assert record["references"] == ["climbmix-a"]
    assert len(record["answer"]) == 1
    assert sum(len(item["text"].split()) for item in record["answer"]) == 900


def test_generation_uses_topic_owned_text_for_shared_docid(tmp_path: Path) -> None:
    _write_handoff(
        tmp_path,
        (
            _generation_topic(
                "rag2026-1", "Question one", [("shared", "Topic one evidence.")]
            ),
            _generation_topic(
                "rag2026-2", "Question two", [("shared", "Topic two evidence.")]
            ),
        ),
    )
    config = load_rag_generation_config(_write_config(tmp_path, _config_text()))
    generator = FakeGenerator(
        {
            "rag2026-1": _topic_output(["shared"], "Topic one answer."),
            "rag2026-2": _topic_output(["shared"], "Topic two answer."),
        }
    )

    asyncio.run(run_generation(config, generator))

    prompts = {call["topic_id"]: call["user_prompt"] for call in generator.calls}
    assert "Topic one evidence." in prompts["rag2026-1"]
    assert "Topic two evidence." not in prompts["rag2026-1"]
    assert "Topic two evidence." in prompts["rag2026-2"]
    assert "Topic one evidence." not in prompts["rag2026-2"]


def test_generation_retries_numeric_citation_once_and_maps_docids_deterministically(
    tmp_path: Path,
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))

    class SequencedGenerator:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def complete_json(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return (
                    {
                        "references": ["climbmix-a"],
                        "answer": [{"text": "Numeric citation.", "citations": [0]}],
                    },
                    {"attempt": 1},
                )
            return (
                {
                    "references": ["climbmix-b"],
                    "answer": [
                        {
                            "text": "The source supports the answer.",
                            "citations": ["climbmix-b", "climbmix-b"],
                        }
                    ],
                },
                {"attempt": 2},
            )

    generator = SequencedGenerator()
    asyncio.run(run_generation(config, generator))

    assert len(generator.calls) == 2
    assert generator.calls[0]["response_schema"]["properties"]["answer"]["items"][
        "properties"
    ]["citations"]["items"] == {"type": "string", "minLength": 1}
    assert "Previous completion failed local validation" in generator.calls[1]["user_prompt"]
    raw_paths = sorted((config.work_dir / "raw").glob("*.json"))
    assert [path.name for path in raw_paths] == [
        f"{competition_rag._safe_topic_name('rag2026-1')}.attempt-1.json",
        f"{competition_rag._safe_topic_name('rag2026-1')}.attempt-2.json",
    ]
    assert [json.loads(path.read_text())["attempt"] for path in raw_paths] == [1, 2]
    row = json.loads(config.output_path.read_text(encoding="utf-8"))
    assert row["references"] == ["climbmix-b"]
    assert row["answer"][0]["citations"] == [0]


@pytest.mark.parametrize(
    ("hint_text", "answer_text"),
    [
        (
            "NIL reforms may not fully address exploitation and inequality in the sports industry",
            "NIL reforms may not fully address exploitation and inequality in the sports industry.",
        ),
        (
            "Sports commercialization has cultural effects, including influencing national identity and creating legacy projects.",
            "sports commercialization has cultural effects, including influencing national identity and creating legacy projects.",
        ),
        (
            "Media and advertising use sports stars to promote products, integrating sports into consumer culture.",
            "Media and  advertising use sports stars to promote products, integrating sports into consumer culture.",
        ),
        (
            "Commercialization can lead to negative cultural impacts, such as obesity and healthcare issues in the US.",
            "Commercialization can lead to negative cultural impacts, such as obesity and healthcare issues in the US.",
        ),
    ],
)
def test_generation_rejects_exact_hint_claim_citing_unlinked_evidence(
    tmp_path: Path,
    hint_text: str,
    answer_text: str,
) -> None:
    topic = _generation_topic(
        "rag2026-1",
        "Question one",
        [("climbmix-a", "Linked evidence."), ("climbmix-b", "Unlinked evidence.")],
    )
    topic = replace(
        topic,
        claim_hints=(replace(topic.claim_hints[0], text=hint_text),),
    )
    _write_handoff(tmp_path, (topic,))
    config = load_rag_generation_config(_write_config(tmp_path, _config_text()))
    generated = {
        "references": ["climbmix-b"],
        "answer": [{"text": answer_text, "citations": ["climbmix-b"]}],
    }
    generator = FakeGenerator({"rag2026-1": generated})

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    assert len(generator.calls) == 2
    assert "cite only docids from that hint's listed" in generator.calls[1]["user_prompt"]
    assert not config.output_path.exists()
    error_path = (
        config.work_dir
        / "errors"
        / f"{competition_rag._safe_topic_name('rag2026-1')}.txt"
    )
    error = error_path.read_text(encoding="utf-8")
    assert "rag2026-1-claim-1" in error
    assert "climbmix-b" in error
    assert "climbmix-a" in error


def test_generation_skips_ambiguous_exact_hint_matches(tmp_path: Path) -> None:
    topic = _generation_topic(
        "rag2026-1",
        "Question one",
        [("climbmix-a", "Evidence A."), ("climbmix-b", "Evidence B.")],
    )
    topic = replace(
        topic,
        claim_hints=(
            replace(
                topic.claim_hints[0],
                claim_id="claim-a",
                text="The shared claim",
                evidence_ids=(topic.evidence[0].evidence_id,),
            ),
            replace(
                topic.claim_hints[0],
                claim_id="claim-b",
                text="the shared claim.",
                evidence_ids=(topic.evidence[1].evidence_id,),
            ),
        ),
    )
    _write_handoff(tmp_path, (topic,))
    config = load_rag_generation_config(_write_config(tmp_path, _config_text()))
    generated = {
        "references": ["climbmix-b"],
        "answer": [{"text": "The shared claim.", "citations": ["climbmix-b"]}],
    }

    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    record = json.loads(config.output_path.read_text(encoding="utf-8"))
    assert record["references"] == ["climbmix-b"]


def test_generation_accepts_exact_hint_claim_citing_only_linked_evidence(
    tmp_path: Path,
) -> None:
    topic = _generation_topic(
        "rag2026-1",
        "Question one",
        [("climbmix-a", "Linked evidence."), ("climbmix-b", "Other evidence.")],
    )
    _write_handoff(tmp_path, (topic,))
    config = load_rag_generation_config(_write_config(tmp_path, _config_text()))
    generated = {
        "references": ["climbmix-a"],
        "answer": [
            {
                "text": "Evidence was selected for rag2026-1.",
                "citations": ["climbmix-a"],
            }
        ],
    }
    generator = FakeGenerator({"rag2026-1": generated})

    asyncio.run(run_generation(config, generator))

    assert len(generator.calls) == 1
    record = json.loads(config.output_path.read_text(encoding="utf-8"))
    assert record["references"] == ["climbmix-a"]


def test_exact_hint_citation_error_reports_every_mismatched_claim() -> None:
    topic = _generation_topic(
        "rag2026-1",
        "Question one",
        [
            ("climbmix-a", "Evidence A."),
            ("climbmix-b", "Evidence B."),
            ("climbmix-c", "Unlinked evidence."),
        ],
    )
    topic = replace(
        topic,
        claim_hints=(
            replace(
                topic.claim_hints[0],
                claim_id="claim-a",
                text="Claim A.",
                evidence_ids=(topic.evidence[0].evidence_id,),
            ),
            replace(
                topic.claim_hints[0],
                claim_id="claim-b",
                text="Claim B.",
                evidence_ids=(topic.evidence[1].evidence_id,),
            ),
        ),
    )
    record = {
        "references": ["climbmix-c"],
        "answer": [
            {"text": "Claim A.", "citations": [0]},
            {"text": "Claim B.", "citations": [0]},
        ],
    }

    with pytest.raises(ValueError, match="exact claim-hint citation mismatches") as error:
        competition_rag._validate_exact_hint_citations(record, topic=topic)

    assert "claim-a" in str(error.value)
    assert "claim-b" in str(error.value)


def test_generation_exhausts_exactly_two_semantic_attempts_without_publishing(
    tmp_path: Path,
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    invalid = {
        "references": ["climbmix-a"],
        "answer": [{"text": "Numeric citation.", "citations": [0]}],
    }
    generator = FakeGenerator({"rag2026-1": invalid})

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    assert len(generator.calls) == 2
    assert not config.output_path.exists()
    assert len(list((config.work_dir / "raw").glob("*.json"))) == 2
    assert not list((config.work_dir / "rows").glob("*.json"))


def test_resume_continues_raw_attempt_numbers_after_semantic_exhaustion(
    tmp_path: Path,
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    invalid = {
        "references": ["climbmix-a"],
        "answer": [{"text": "Numeric citation.", "citations": [0]}],
    }
    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": invalid})))

    valid = _topic_output(["climbmix-a"], "The source supports the answer.")
    resumed = FakeGenerator({"rag2026-1": valid})
    asyncio.run(run_generation(replace(config, resume=True), resumed))

    assert len(resumed.calls) == 1
    raw_paths = sorted((config.work_dir / "raw").glob("*.json"))
    topic_name = competition_rag._safe_topic_name("rag2026-1")
    assert [path.name for path in raw_paths] == [
        f"{topic_name}.attempt-1.json",
        f"{topic_name}.attempt-2.json",
        f"{topic_name}.attempt-3.json",
    ]


def test_generation_consumes_only_the_selected_evidence_manifest_contract(
    tmp_path: Path,
) -> None:
    _write_handoff(
        tmp_path,
        (
            _generation_topic(
                "rag2026-9",
                "Unselected official question",
                [("climbmix-9a", "Unselected evidence.")],
            ),
            _generation_topic(
                "rag2026-3",
                "Second selected official question",
                [
                    ("climbmix-3a", "First selected passage."),
                    ("climbmix-3b", "Second selected passage."),
                ],
            ),
            _generation_topic(
                "rag2026-4",
                "First selected official question",
                [("climbmix-4a", "Third selected passage.")],
            ),
        ),
    )
    config = load_rag_generation_config(
        _write_config(
            tmp_path,
            _config_text(
                experiment_extra="  topic_ids: [rag2026-4, rag2026-3]\n",
            ),
        )
    )
    generator = FakeGenerator(
        {
            "rag2026-3": _topic_output(
                ["climbmix-3a", "climbmix-3b"], "Both selected documents support it."
            ),
            "rag2026-4": _topic_output(
                ["climbmix-4a"], "The selected document supports it."
            ),
        }
    )

    asyncio.run(run_generation(config, generator))

    assert config.output_path.name == "rag_output_trec_rag_2026.jsonl"
    rows = [
        json.loads(line)
        for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["metadata"]["narrative_id"] for row in rows] == [
        "rag2026-3",
        "rag2026-4",
    ]
    assert [list(row) for row in rows] == [
        ["metadata", "references", "answer"],
        ["metadata", "references", "answer"],
    ]
    assert [list(row["metadata"]) for row in rows] == [
        ["team_id", "narrative_id", "narrative", "run_id", "run_desc"],
        ["team_id", "narrative_id", "narrative", "run_id", "run_desc"],
    ]
    assert [row["references"] for row in rows] == [
        ["climbmix-3a", "climbmix-3b"],
        ["climbmix-4a"],
    ]
    assert [row["answer"][0]["citations"] for row in rows] == [[0, 1], [0]]
    prompts = {call["topic_id"]: call["user_prompt"] for call in generator.calls}
    assert "First selected passage." in prompts["rag2026-3"]
    assert "Third selected passage." in prompts["rag2026-4"]
    assert "Unselected evidence." not in "\n".join(prompts.values())


def test_resume_reuses_valid_rows_without_model_calls(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    first = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "A supports it."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )
    asyncio.run(run_generation(config, first))

    resumed = FakeGenerator({})
    asyncio.run(run_generation(replace(config, resume=True), resumed))

    assert resumed.calls == []
    assert config.output_path.exists()


def test_resume_regenerates_row_with_non_strict_metadata(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    initial = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "A supports it."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )
    asyncio.run(run_generation(config, initial))
    row_path = next(
        path
        for path in (config.work_dir / "rows").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["metadata"]["narrative_id"]
        == "rag2026-1"
    )
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["metadata"]["participant_note"] = "organizer-valid but not generated-profile"
    row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    resumed = FakeGenerator(
        {"rag2026-1": _topic_output(["climbmix-a"], "Regenerated strict row.")}
    )

    asyncio.run(run_generation(replace(config, resume=True), resumed))

    assert [call["topic_id"] for call in resumed.calls] == ["rag2026-1"]
    output_rows = [
        json.loads(line)
        for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]
    regenerated = next(
        item for item in output_rows if item["metadata"]["narrative_id"] == "rag2026-1"
    )
    assert set(regenerated["metadata"]) == {
        "team_id",
        "narrative_id",
        "narrative",
        "run_id",
        "run_desc",
    }


def test_generation_limits_parallel_model_calls_to_configured_concurrency(
    tmp_path: Path,
) -> None:
    config = replace(_pipeline_config(tmp_path), concurrency=1)

    class CountingGenerator(FakeGenerator):
        def __init__(self) -> None:
            super().__init__(
                {
                    "rag2026-1": _topic_output(["climbmix-a"], "A supports it."),
                    "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
                }
            )
            self._lock = threading.Lock()
            self.active = 0
            self.maximum_active = 0

        def complete_json(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            with self._lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                time.sleep(0.02)
                return super().complete_json(**kwargs)
            finally:
                with self._lock:
                    self.active -= 1

    generator = CountingGenerator()
    asyncio.run(run_generation(config, generator))

    assert generator.maximum_active == 1


def test_overwrite_clears_only_generation_state_before_failure_then_resume_regenerates(
    tmp_path: Path,
) -> None:
    config = _pipeline_config(tmp_path)
    initial = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "Old one."),
            "rag2026-2": _topic_output(["climbmix-c"], "Old two."),
        }
    )
    asyncio.run(run_generation(config, initial))
    handoff_bytes = config.handoff_manifest_path.read_bytes()

    with pytest.raises(RuntimeError, match="2 topic"):
        asyncio.run(run_generation(replace(config, overwrite=True), FakeGenerator({})))

    assert not config.output_path.exists()
    assert handoff_bytes == config.handoff_manifest_path.read_bytes()

    resumed = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "Fresh one."),
            "rag2026-2": _topic_output(["climbmix-c"], "Fresh two."),
        }
    )
    asyncio.run(run_generation(replace(config, resume=True), resumed))

    assert {call["topic_id"] for call in resumed.calls} == {"rag2026-1", "rag2026-2"}
    assert "Old one." not in config.output_path.read_text(encoding="utf-8")
    assert "Fresh one." in config.output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "overlap",
    [
        "manifest_equal_output",
        "manifest_inside_work",
        "manifest_contains_work",
        "manifest_contains_output",
    ],
)
def test_overwrite_rejects_input_and_deletion_target_overlap_before_deleting(
    tmp_path: Path, overlap: str
) -> None:
    config = _pipeline_config(tmp_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text("old output\n", encoding="utf-8")
    config.work_dir.mkdir(parents=True, exist_ok=True)
    sentinel = config.work_dir / "old-row.json"
    sentinel.write_text("old row\n", encoding="utf-8")
    original_manifest = config.handoff_manifest_path.read_bytes()

    if overlap == "manifest_equal_output":
        config = replace(
            config,
            output_path=config.handoff_manifest_path,
            overwrite=True,
        )
    elif overlap == "manifest_inside_work":
        nested_manifest = config.work_dir / "manifest.json"
        nested_manifest.write_bytes(original_manifest)
        config = replace(
            config,
            handoff_manifest_path=nested_manifest,
            overwrite=True,
        )
    elif overlap == "manifest_contains_work":
        container = tmp_path / "input-container"
        container.mkdir()
        nested_work = container / "work"
        nested_work.mkdir()
        (nested_work / "old-row.json").write_text("old row\n", encoding="utf-8")
        config = replace(
            config,
            handoff_manifest_path=container,
            work_dir=nested_work,
            overwrite=True,
        )
        sentinel = nested_work / "old-row.json"
    else:
        container = tmp_path / "input-container"
        container.mkdir()
        nested_output = container / "submission.jsonl"
        nested_output.write_text("old output\n", encoding="utf-8")
        config = replace(
            config,
            handoff_manifest_path=container,
            output_path=nested_output,
            overwrite=True,
        )

    with pytest.raises(ValueError, match="overlap"):
        asyncio.run(run_generation(config, FakeGenerator({})))

    assert sentinel.read_text(encoding="utf-8") == "old row\n"
    if overlap not in {"manifest_equal_output", "manifest_inside_work"}:
        assert config.output_path.read_text(encoding="utf-8") == "old output\n"
    if overlap != "manifest_equal_output":
        assert config.handoff_manifest_path != config.output_path
    assert (
        tmp_path
        / "outputs/facet-deepseek-b40-v2/generation_handoff_manifest.json"
    ).read_bytes() == original_manifest


def test_overwrite_validates_handoff_before_clearing_old_state(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    initial = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "Old one."),
            "rag2026-2": _topic_output(["climbmix-c"], "Old two."),
        }
    )
    asyncio.run(run_generation(config, initial))
    previous_output = config.output_path.read_bytes()
    previous_identity = (config.work_dir / "generation_identity.json").read_bytes()
    valid_manifest = config.handoff_manifest_path.read_bytes()
    config.handoff_manifest_path.write_text("not canonical JSON\n", encoding="utf-8")

    with pytest.raises(ValueError):
        asyncio.run(run_generation(replace(config, overwrite=True), FakeGenerator({})))

    assert config.output_path.read_bytes() == previous_output
    assert (config.work_dir / "generation_identity.json").read_bytes() == previous_identity
    config.handoff_manifest_path.write_bytes(valid_manifest)


@pytest.mark.parametrize("artifact", ["raw/response.json", "errors/topic.txt", "scratch.bin"])
def test_create_rejects_any_existing_work_artifact(tmp_path: Path, artifact: str) -> None:
    config = _pipeline_config(tmp_path)
    path = config.work_dir / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("existing artifact\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifacts exist"):
        asyncio.run(run_generation(config, FakeGenerator({})))

    assert path.read_text(encoding="utf-8") == "existing artifact\n"


def test_final_publication_does_not_reuse_predictable_temp_name(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    old_shared_temp = config.output_path.with_name(config.output_path.name + ".tmp")
    old_shared_temp.write_text("unrelated sentinel\n", encoding="utf-8")
    generator = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "A supports it."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )

    asyncio.run(run_generation(config, generator))

    assert old_shared_temp.read_text(encoding="utf-8") == "unrelated sentinel\n"


def test_atomic_publication_fsyncs_file_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pipeline_config(tmp_path)
    synced_kinds: list[str] = []
    real_fsync = os.fsync

    def observing_fsync(file_descriptor: int) -> None:
        mode = os.fstat(file_descriptor).st_mode
        synced_kinds.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(file_descriptor)

    monkeypatch.setattr(competition_rag.os, "fsync", observing_fsync)
    generator = FakeGenerator(
        {
            "rag2026-1": _topic_output(["climbmix-a"], "A supports it."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )

    asyncio.run(run_generation(config, generator))

    assert "file" in synced_kinds
    if os.name == "nt":
        assert "directory" not in synced_kinds
    else:
        assert "directory" in synced_kinds


def test_second_process_cannot_mutate_same_generation_artifacts(tmp_path: Path) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    entered = threading.Event()
    release = threading.Event()
    first_errors: list[BaseException] = []

    class BlockingGenerator(FakeGenerator):
        def complete_json(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test release timed out")
            return super().complete_json(**kwargs)

    first = BlockingGenerator(
        {"rag2026-2": _topic_output(["climbmix-c"], "First process.")}
    )

    def run_first() -> None:
        try:
            asyncio.run(run_generation(config, first))
        except BaseException as exc:
            first_errors.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            asyncio.run(
                run_generation(
                    config,
                    FakeGenerator(
                        {"rag2026-2": _topic_output(["climbmix-c"], "Second process.")}
                    ),
                )
            )
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert first_errors == []


def test_invalid_model_output_keeps_final_submission_absent(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    generator = FakeGenerator(
        {
            "rag2026-1": _topic_output(["not-retrieved"], "Unsupported."),
            "rag2026-2": _topic_output(["climbmix-c"], "C supports it."),
        }
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    assert not config.output_path.exists()
    assert list((config.work_dir / "errors").glob("*.txt"))


@pytest.mark.parametrize(
    ("references", "citations"),
    [
        pytest.param(["climbmix-a"], [], id="empty-citations"),
    ],
)
def test_generation_rejects_organizer_valid_rows_outside_strict_profile(
    tmp_path: Path,
    references: list[str],
    citations: list[int | str],
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": references,
        "answer": [{"text": "Generated answer.", "citations": citations}],
    }

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    assert not config.output_path.exists()


def test_generation_now_prunes_an_uncited_reference_instead_of_failing(tmp_path: Path) -> None:
    """Uncited references used to fail the strict profile; they are normalized away."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a", "climbmix-b"],
        "answer": [
            {"text": "Only the second source is cited.", "citations": ["climbmix-b"]}
        ],
    }

    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    rows = [json.loads(line) for line in config.output_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["references"] == ["climbmix-b"]
    assert rows[0]["answer"][0]["citations"] == [0]
    assert rows[0]["answer"][0]["text"] == "Only the second source is cited."


def test_generation_normalizes_duplicate_citations_end_to_end(tmp_path: Path) -> None:
    """Cover the full generate-to-publish path, not just the normalizer in isolation."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a", "climbmix-b"],
        "answer": [
            {
                "text": "One document, cited twice.",
                "citations": ["climbmix-b", "climbmix-b"],
            }
        ],
    }

    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    rows = [json.loads(line) for line in config.output_path.read_text().splitlines()]
    assert rows[0]["references"] == ["climbmix-b"]
    assert rows[0]["answer"][0]["citations"] == [0]


def test_truncated_completion_is_rejected_rather_than_shortened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grammar-constrained decoding can close the JSON validly while the answer is cut short."""
    generator = competition_rag.OpenRouterJsonGenerator(
        api_base="https://openrouter.test",
        api_key="test-key",
        model="m",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=10,
        timeout_seconds=5.0,
        transport_max_attempts=1,
        http_transport=_mock_http_transport(
            lambda *args, **kwargs: FakeHttpResponse(200, body)
        ),
    )
    body = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": json.dumps(
                        {
                            "references": ["climbmix-a"],
                            "answer": [{"text": "Cut short.", "citations": [0]}],
                        }
                    )
                },
            }
        ]
    }
    with pytest.raises(competition_rag.SemanticCompletionError, match="truncated"):
        generator.complete_json(
            topic_id="rag2026-1",
            system_prompt="s",
            user_prompt="u",
            response_schema=competition_rag.output_schema(),
        )


@pytest.mark.parametrize(
    "finish_reason",
    [
        pytest.param("error", id="provider-error-with-partial-content"),
        pytest.param("content_filter", id="content-filter"),
        pytest.param("tool_calls", id="tool-calls"),
        pytest.param(None, id="absent"),
        pytest.param("unrecognised", id="unknown"),
    ],
)
def test_only_a_stop_finish_reason_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish_reason: str | None
) -> None:
    """OpenRouter returns HTTP 200 with finish_reason 'error' and partial content."""
    generator = competition_rag.OpenRouterJsonGenerator(
        api_base="https://openrouter.test",
        api_key="test-key",
        model="m",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=100,
        timeout_seconds=5.0,
        transport_max_attempts=1,
        http_transport=_mock_http_transport(
            lambda *args, **kwargs: FakeHttpResponse(200, {"choices": [choice]})
        ),
    )
    choice: dict[str, Any] = {
        "message": {
            "content": json.dumps(
                {
                    "references": ["climbmix-a"],
                    "answer": [{"text": "Looks complete.", "citations": [0]}],
                }
            )
        }
    }
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    with pytest.raises(competition_rag.SemanticCompletionError, match="rather than 'stop'"):
        generator.complete_json(
            topic_id="rag2026-1",
            system_prompt="s",
            user_prompt="u",
            response_schema=competition_rag.output_schema(),
        )


def test_resume_refuses_rows_generated_under_different_settings(tmp_path: Path) -> None:
    """Resume reuses saved rows; shape revalidation cannot detect a changed model or prompt."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": ["climbmix-a"]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    changed = replace(config, resume=True, reasoning_effort="high")
    with pytest.raises(ValueError, match="different settings"):
        asyncio.run(run_generation(changed, FakeGenerator({"rag2026-1": generated})))


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"max_tokens": 7000}, id="max-tokens"),
        pytest.param({"temperature": 0.2}, id="temperature"),
        pytest.param({"api_base": "https://elsewhere.test"}, id="api-base"),
        pytest.param({"model": "other/model"}, id="model"),
        pytest.param({"provider": "other-provider"}, id="provider"),
        pytest.param({"reasoning_effort": "high"}, id="reasoning-effort"),
        pytest.param({"structured_output": "json_object"}, id="structured-output"),
    ],
)
def test_resume_refuses_any_request_affecting_change(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    """Every setting that reaches the request must invalidate a resume."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": ["climbmix-a"]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    with pytest.raises(ValueError, match="different settings"):
        asyncio.run(
            run_generation(
                replace(config, resume=True, **change), FakeGenerator({"rag2026-1": generated})
            )
        )


def test_resume_refuses_a_changed_valid_handoff(tmp_path: Path) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = _topic_output(["climbmix-a"], "A grounded claim.")
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    changed_handoff = GenerationHandoff(
        producer=HandoffProducer(
            source_contract=SOURCE_CONTRACT,
            retrieval_run_id="facet-deepseek-b40-v2",
            producer_revision="changed-test-revision",
        ),
        topics=(
            _generation_topic(
                "rag2026-2", "Question two", [("climbmix-c", "Evidence C.")]
            ),
            _generation_topic(
                "rag2026-1",
                "Question one",
                [("climbmix-a", "Changed selected evidence.")],
            ),
        ),
    )
    config.handoff_manifest_path.write_bytes(
        serialize_generation_handoff(changed_handoff)
    )

    with pytest.raises(ValueError, match="different settings"):
        asyncio.run(run_generation(replace(config, resume=True), FakeGenerator({})))


def test_resume_refuses_a_changed_selected_topic_set(tmp_path: Path) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    asyncio.run(
        run_generation(
            config,
            FakeGenerator(
                {"rag2026-1": _topic_output(["climbmix-a"], "A grounded claim.")}
            ),
        )
    )

    with pytest.raises(ValueError, match="different settings"):
        asyncio.run(
            run_generation(
                replace(config, resume=True, topic_ids=("rag2026-2",)),
                FakeGenerator({}),
            )
        )


@pytest.mark.parametrize("change", ["contract", "rendered-prompt"])
def test_resume_refuses_a_changed_prompt_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    asyncio.run(
        run_generation(
            config,
            FakeGenerator(
                {"rag2026-1": _topic_output(["climbmix-a"], "A grounded claim.")}
            ),
        )
    )

    if change == "contract":
        monkeypatch.setattr(
            competition_rag,
            "PROMPT_CONTRACT_VERSION",
            "selected_evidence_one_shot_changed",
        )
    else:
        original_render_prompt = competition_rag.render_prompt
        monkeypatch.setattr(
            competition_rag,
            "render_prompt",
            lambda topic: original_render_prompt(topic) + "\nChanged writer instruction.",
        )

    with pytest.raises(ValueError, match="different settings"):
        asyncio.run(run_generation(replace(config, resume=True), FakeGenerator({})))


def test_generation_identity_covers_every_request_setting(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    handoff = load_generation_handoff(config.handoff_manifest_path)
    topics = select_generation_topics(handoff, config.topic_ids)
    base = competition_rag._generation_identity(config, handoff, topics)

    for field, value in [
        ("temperature", 0.2),
        ("max_tokens", 7000),
        ("api_base", "https://elsewhere.test"),
        ("structured_output", "json_object"),
        ("reasoning_effort", "high"),
        ("strategy", "coverage_aware"),
        ("provider", "other-provider"),
        ("model", "other/model"),
    ]:
        changed = competition_rag._generation_identity(
            replace(config, **{field: value}),
            handoff,
            topics,
        )
        assert changed != base, f"{field} does not invalidate the identity"


def test_identity_version_rejects_rows_written_before_generation_strategy(
    tmp_path: Path,
) -> None:
    """Version 6 rows did not bind the config-controlled generation strategy."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": ["climbmix-a"]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))
    path = config.work_dir / "generation_identity.json"
    recorded = json.loads(path.read_text())
    assert recorded["identity_version"] == 7
    assert recorded["citation_validation_contract_version"] == (
        "exact_hint_linked_docids_v1"
    )
    assert recorded["call_telemetry_contract_version"] == (
        "generation_call_telemetry_v1"
    )
    recorded["identity_version"] = 6
    path.write_text(json.dumps(recorded), encoding="utf-8")

    with pytest.raises(ValueError, match="older revision"):
        asyncio.run(run_generation(replace(config, resume=True), FakeGenerator({})))


def test_resume_refuses_rows_that_predate_settings_tracking(tmp_path: Path) -> None:
    """A workdir from before identity tracking cannot be shown to match; adopt nothing."""
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": ["climbmix-a"]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))
    (config.work_dir / "generation_identity.json").unlink()

    with pytest.raises(ValueError, match="predate settings tracking"):
        asyncio.run(run_generation(replace(config, resume=True), FakeGenerator({})))


def test_resume_refuses_an_older_identity_version(tmp_path: Path) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": ["climbmix-a"]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))
    path = config.work_dir / "generation_identity.json"
    recorded = json.loads(path.read_text())
    recorded["identity_version"] = 1
    path.write_text(json.dumps(recorded), encoding="utf-8")

    with pytest.raises(ValueError, match="older revision"):
        asyncio.run(run_generation(replace(config, resume=True), FakeGenerator({})))


def test_resume_accepts_rows_generated_under_the_same_settings(tmp_path: Path) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    generated = {
        "references": ["climbmix-a"],
        "answer": [{"text": "A grounded claim.", "citations": ["climbmix-a"]}],
    }
    asyncio.run(run_generation(config, FakeGenerator({"rag2026-1": generated})))

    resumed = FakeGenerator({})
    asyncio.run(run_generation(replace(config, resume=True), resumed))
    assert resumed.calls == [], "resume should reuse the saved row without regenerating"


def test_persisted_raw_responses_and_errors_redact_configured_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pipeline_config(tmp_path)
    api_key = "reflected-secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)

    class ReflectingGenerator:
        def complete_json(
            self, *, topic_id: str, **kwargs: Any
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            del kwargs
            if topic_id == "rag2026-1":
                return (
                    _topic_output(["climbmix-a"], "A supports it."),
                    {
                        "authorization": f"Bearer {api_key}",
                        "nested": [f"prefix-{api_key}-suffix"],
                    },
                )
            raise ValueError(f"provider reflected {api_key} in an error")

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, ReflectingGenerator()))

    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in config.work_dir.rglob("*")
        if path.is_file()
    )
    assert api_key not in persisted
    assert "[REDACTED]" in persisted


def test_openrouter_redacts_its_key_from_persisted_success_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "client-only-secret-token"
    monkeypatch.delenv(config.api_key_env, raising=False)
    response = FakeHttpResponse(
        200,
        {
            "id": f"provider-reflected-{api_key}",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            _topic_output(["climbmix-c"], "C supports it.")
                        )
                    },
                }
            ],
        },
    )
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
        http_transport=_mock_http_transport(lambda *args, **kwargs: response),
    )

    asyncio.run(run_generation(config, generator))

    raw_path = next((config.work_dir / "raw").glob("*.json"))
    persisted = raw_path.read_text(encoding="utf-8")
    assert api_key not in persisted
    assert "provider-reflected-[REDACTED]" in persisted


@pytest.mark.parametrize(
    ("api_key", "reflected_key"),
    [
        ("secret-token", "secret%2Dtoken"),
        ("secret-token", "secret%252Dtoken"),
        pytest.param(
            "secret-token",
            r"secret%5Cu002Dtoken",
            id="percent-then-unicode-escape",
        ),
        pytest.param(
            "secret-token",
            r"secret\u00252Dtoken",
            id="unicode-escape-then-percent",
        ),
        pytest.param(
            "secret%2Dtoken",
            "secret%252Dtoken",
            id="configured-secret-contains-percent-escape",
        ),
    ],
)
def test_percent_encoded_api_key_in_parsed_envelope_never_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api_key: str,
    reflected_key: str,
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    response = FakeHttpResponse(
        200,
        {
            "id": f"provider-reflected-{reflected_key}",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            _topic_output(["climbmix-c"], "C supports it.")
                        )
                    },
                }
            ],
        },
    )
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
        http_transport=_mock_http_transport(lambda *args, **kwargs: response),
    )

    asyncio.run(run_generation(config, generator))

    raw = json.loads(next((config.work_dir / "raw").glob("*.json")).read_text(encoding="utf-8"))
    assert raw["id"] == "[REDACTED]"
    persisted = json.dumps(raw)
    assert api_key not in persisted
    assert reflected_key not in persisted


class FakeHttpResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers: dict[str, str] = {}
        self.text = json.dumps(payload)

    def json(self) -> object:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _without_call_telemetry(raw: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(raw)
    telemetry = cleaned.pop("_trec_rag_call")
    assert telemetry["schema_version"] == "generation_call_telemetry_v1"
    assert telemetry["provider"] == "openrouter"
    assert telemetry["latency_ms"] >= 0
    assert telemetry["transport_attempts"] >= 1
    assert telemetry["transport_retries"] == telemetry["transport_attempts"] - 1
    return cleaned


def _mock_http_transport(post: Any) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        response = post(
            str(request.url),
            headers=dict(request.headers),
            json=json.loads(request.content),
        )
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.text.encode("utf-8"),
        )

    return httpx.MockTransport(handle)


def _openrouter_generator(
    post: Any | None = None,
    *,
    timeout_seconds: float = 30,
    transport_max_attempts: int = 3,
) -> OpenRouterJsonGenerator:
    if post is None:
        def post(*args: Any, **kwargs: Any) -> FakeHttpResponse:
            return _provider_success()

    return OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key="secret",
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=timeout_seconds,
        transport_max_attempts=transport_max_attempts,
        http_transport=_mock_http_transport(post),
    )


def _provider_success() -> FakeHttpResponse:
    return FakeHttpResponse(
        200,
        {
            "id": "response-1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(_topic_output(["climbmix-a"], "Supported."))
                    },
                }
            ],
        },
    )


def test_openrouter_uses_an_openai_compatible_http_transport() -> None:
    captured: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_provider_success().payload)

    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key="secret",
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=3,
        http_transport=httpx.MockTransport(handle),
    )

    generated, raw = generator.complete_json(
        topic_id="rag2026-1",
        system_prompt="system",
        user_prompt="user",
        response_schema={"type": "object"},
    )

    assert captured["url"] == "https://openrouter.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["body"]["reasoning"] == {"effort": "medium", "exclude": True}
    assert captured["body"]["usage"] == {"include": True}
    assert captured["body"]["provider"] == {"require_parameters": True}
    assert generated["answer"][0]["citations"] == ["climbmix-a"]
    assert raw["id"] == "response-1"
    assert raw["_trec_rag_call"]["transport_attempts"] == 1
    assert raw["_trec_rag_call"]["transport_retries"] == 0
    assert raw["_trec_rag_call"]["latency_ms"] >= 0


def test_openrouter_honors_an_http_date_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [FakeHttpResponse(429, {"error": "slow down"}), _provider_success()]
    responses[0].headers = {"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}
    sleeps: list[float] = []
    monkeypatch.setattr(httpx_retry_module.time, "sleep", sleeps.append)

    _, raw = _openrouter_generator(
        lambda *args, **kwargs: responses.pop(0), timeout_seconds=900
    ).complete_json(
        topic_id="rag2026-1",
        system_prompt="system",
        user_prompt="user",
        response_schema={"type": "object"},
    )

    assert sleeps == [900.0]
    assert raw["_trec_rag_call"]["transport_attempts"] == 2
    assert raw["_trec_rag_call"]["transport_retries"] == 1


def test_openrouter_request_uses_strict_schema_and_medium_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        captured.update({"url": url, **kwargs})
        return _provider_success()

    generated, raw = _openrouter_generator(fake_post).complete_json(
        topic_id="rag2026-1", system_prompt="system", user_prompt="user", response_schema={"type": "object"}
    )

    assert captured["url"] == "https://openrouter.example/v1/chat/completions"
    assert captured["json"]["reasoning"] == {"effort": "medium", "exclude": True}
    assert captured["json"]["usage"] == {"include": True}
    assert captured["json"]["provider"] == {"require_parameters": True}
    assert captured["json"]["response_format"]["type"] == "json_schema"
    assert captured["json"]["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in captured["json"]
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert generated["answer"][0]["citations"] == ["climbmix-a"]
    assert raw["id"] == "response-1"


def test_malformed_semantic_completion_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        nonlocal calls
        del url, kwargs
        calls += 1
        return FakeHttpResponse(200, {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]})

    with pytest.raises(ValueError, match="no repair call") as error:
        _openrouter_generator(fake_post).complete_json(
            topic_id="rag2026-1", system_prompt="system", user_prompt="user", response_schema={"type": "object"}
        )

    assert calls == 1
    assert error.value.raw_response["choices"][0]["message"]["content"] == "not JSON"


def test_http_200_non_json_gets_one_fresh_semantic_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))

    class NonJsonResponse(FakeHttpResponse):
        def json(self) -> object:
            raise ValueError("not JSON")

    responses: list[FakeHttpResponse] = [NonJsonResponse(200, None), _provider_success()]
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        nonlocal calls
        del url, kwargs
        calls += 1
        return responses.pop(0)

    asyncio.run(run_generation(config, _openrouter_generator(fake_post)))

    assert calls == 2
    assert len(list((config.work_dir / "raw").glob("*.failed.json"))) == 1
    assert len(list((config.work_dir / "raw").glob("*.json"))) == 2
    assert config.output_path.exists()


def test_missing_finish_reason_gets_one_fresh_semantic_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    responses: list[FakeHttpResponse] = [
        FakeHttpResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(_topic_output(["climbmix-a"], "Partial."))
                        }
                    }
                ]
            },
        ),
        _provider_success(),
    ]
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        nonlocal calls
        del url, kwargs
        calls += 1
        return responses.pop(0)

    asyncio.run(run_generation(config, _openrouter_generator(fake_post)))

    assert calls == 2
    assert config.output_path.exists()


def test_missing_finish_reason_exhausts_two_semantic_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    response = FakeHttpResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(_topic_output(["climbmix-a"], "Partial."))
                    }
                }
            ]
        },
    )
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        nonlocal calls
        del url, kwargs
        calls += 1
        return response

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, _openrouter_generator(fake_post)))

    assert calls == 2
    assert len(list((config.work_dir / "raw").glob("*.failed.json"))) == 2
    assert not config.output_path.exists()


def test_semantic_retry_does_not_repeat_truncation_or_exhausted_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-1",))
    calls = 0

    def truncated_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        nonlocal calls
        del url, kwargs
        calls += 1
        return FakeHttpResponse(
            200,
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": json.dumps(_topic_output(["climbmix-a"], "Cut short."))
                        },
                    }
                ]
            },
        )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, _openrouter_generator(truncated_post)))
    assert calls == 1

    calls = 0

    def failed_transport(url: str, **kwargs: Any) -> FakeHttpResponse:
        nonlocal calls
        del url, kwargs
        calls += 1
        raise httpx.ConnectError("connection lost")

    monkeypatch.setattr(httpx_retry_module.time, "sleep", lambda _: None)
    other = replace(_pipeline_config(tmp_path / "transport"), topic_ids=("rag2026-1",))
    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(other, _openrouter_generator(failed_transport)))
    assert calls == 3


def test_transient_retries_repeat_the_identical_request(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []
    responses = [FakeHttpResponse(500, {"error": "temporary"}), _provider_success()]

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        del url
        bodies.append(copy.deepcopy(kwargs["json"]))
        return responses.pop(0)

    monkeypatch.setattr(httpx_retry_module.time, "sleep", lambda _: None)
    _openrouter_generator(fake_post).complete_json(
        topic_id="rag2026-1", system_prompt="system", user_prompt="user", response_schema={"type": "object"}
    )

    assert len(bodies) == 2
    assert bodies[0] == bodies[1]


def test_openrouter_honors_retry_after_longer_than_thirty_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [FakeHttpResponse(429, {"error": "slow down"}), _provider_success()]
    responses[0].headers = {"Retry-After": "120"}
    sleeps: list[float] = []

    monkeypatch.setattr(httpx_retry_module.time, "sleep", sleeps.append)
    generator = _openrouter_generator(
        lambda *args, **kwargs: responses.pop(0), timeout_seconds=900
    )

    generator.complete_json(
        topic_id="rag2026-1",
        system_prompt="system",
        user_prompt="user",
        response_schema={"type": "object"},
    )

    assert sleeps == [120.0]


@pytest.mark.parametrize("retry_after", ["nan", "NaN", "inf", "-inf", "1e400"])
def test_openrouter_falls_back_for_nonfinite_retry_after(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: str,
) -> None:
    responses = [FakeHttpResponse(429, {"error": "slow down"}), _provider_success()]
    responses[0].headers = {"Retry-After": retry_after}
    sleeps: list[float] = []

    monkeypatch.setattr(httpx_retry_module.time, "sleep", sleeps.append)

    _openrouter_generator(lambda *args, **kwargs: responses.pop(0)).complete_json(
        topic_id="rag2026-1",
        system_prompt="system",
        user_prompt="user",
        response_schema={"type": "object"},
    )

    assert sleeps == [1.0]


@pytest.mark.parametrize("status_code", [400, 429, 503])
def test_http_failures_persist_sanitized_status_body_and_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "reflected-secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)
    payload = {"error": {"message": f"rejected bearer {api_key}"}}

    def fake_post(url: str, **kwargs: Any) -> FakeHttpResponse:
        del url, kwargs
        return FakeHttpResponse(status_code, payload)

    monkeypatch.setattr(httpx_retry_module.time, "sleep", lambda _: None)
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
        http_transport=_mock_http_transport(fake_post),
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    raw = json.loads(failed_raw.read_text(encoding="utf-8"))
    assert raw["http_status"] == status_code
    assert raw["envelope"] == {"error": {"message": "rejected bearer [REDACTED]"}}
    assert api_key not in failed_raw.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "escaped_key",
    [
        r"\u0073ecret-token",
        "".join(f"\\u{ord(character):04X}" for character in "secret-token"),
    ],
)
def test_http_json_failure_omits_body_with_decodable_escaped_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, escaped_key: str
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)
    response = FakeHttpResponse(
        400,
        {"error": {"message": f"rejected bearer {api_key}"}},
    )
    response.text = '{"error":{"message":"rejected bearer ' + escaped_key + '"}}'
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
        http_transport=_mock_http_transport(lambda *args, **kwargs: response),
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    raw = json.loads(failed_raw.read_text(encoding="utf-8"))
    assert _without_call_telemetry(raw) == {
        "http_status": 400,
        "envelope": {"error": {"message": "rejected bearer [REDACTED]"}},
    }
    assert api_key not in json.dumps(raw)


def test_http_non_json_failure_omits_body_and_keeps_non_reversible_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)

    class NonJsonResponse(FakeHttpResponse):
        def json(self) -> object:
            raise ValueError("not JSON")

    response = NonJsonResponse(400, None)
    body_text = r"gateway rejected \u0073ecret-token"
    response.text = body_text
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
        http_transport=_mock_http_transport(lambda *args, **kwargs: response),
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    raw = json.loads(failed_raw.read_text(encoding="utf-8"))
    body_bytes = body_text.encode("utf-8")
    assert _without_call_telemetry(raw) == {
        "http_status": 400,
        "body_omitted": True,
        "body_utf8_byte_length": len(body_bytes),
        "body_utf8_sha256": sha256(body_bytes).hexdigest(),
    }


def test_http_non_json_failure_omits_plain_body_and_keeps_non_reversible_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)

    class NonJsonResponse(FakeHttpResponse):
        def json(self) -> object:
            raise ValueError("not JSON")

    response = NonJsonResponse(400, None)
    body_text = f"gateway rejected bearer {api_key}"
    response.text = body_text
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
        http_transport=_mock_http_transport(lambda *args, **kwargs: response),
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    persisted = failed_raw.read_text(encoding="utf-8")
    body_bytes = body_text.encode("utf-8")
    assert _without_call_telemetry(json.loads(persisted)) == {
        "http_status": 400,
        "body_omitted": True,
        "body_utf8_byte_length": len(body_bytes),
        "body_utf8_sha256": sha256(body_bytes).hexdigest(),
    }
    assert api_key not in persisted
    assert body_text not in persisted
def test_http_non_json_success_never_persists_nested_decodable_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_pipeline_config(tmp_path), topic_ids=("rag2026-2",))
    api_key = "secret-token"
    monkeypatch.setenv(config.api_key_env, api_key)

    class NonJsonResponse(FakeHttpResponse):
        def json(self) -> object:
            raise ValueError("not JSON")

    response = NonJsonResponse(200, None)
    body_text = r"gateway reflected \u005cu0073ecret-token"
    response.text = body_text
    generator = OpenRouterJsonGenerator(
        api_base="https://openrouter.example/v1",
        api_key=api_key,
        model="openai/gpt-5.6-sol",
        reasoning_effort="medium",
        temperature=None,
        max_tokens=6000,
        timeout_seconds=30,
        transport_max_attempts=2,
        http_transport=_mock_http_transport(lambda *args, **kwargs: response),
    )

    with pytest.raises(RuntimeError, match="1 topic"):
        asyncio.run(run_generation(config, generator))

    failed_raw = next((config.work_dir / "raw").glob("*.failed.json"))
    persisted = failed_raw.read_text(encoding="utf-8")
    body_bytes = body_text.encode("utf-8")
    assert _without_call_telemetry(json.loads(persisted)) == {
        "http_status": 200,
        "body_omitted": True,
        "body_utf8_byte_length": len(body_bytes),
        "body_utf8_sha256": sha256(body_bytes).hexdigest(),
    }
    assert api_key not in persisted
    assert body_text not in persisted
