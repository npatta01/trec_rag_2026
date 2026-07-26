"""Regenerate the full-evidence Topic 213 report in the TREC RAG 2026 format."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from trec_rag.repo_env import find_repo_root, load_repo_env
from trec_rag.topic213_ragnarok_experiment import SpacySentenceTokenizer
from trec_rag.topic213_response_experiment import (
    OpenAICompatibleJsonClient,
    _audit_section,
    _complete_validated,
    _require_nonempty_string,
    _sha256,
    _write_json,
    _write_jsonl,
    compute_evaluation_metrics,
    evaluate_nuggets,
    load_nuggets,
)
from trec_rag.topic213_utokyo_experiment import normalize_answer_label


SCHEMA_VERSION = "topic213-trec-rag-2026-submission-v1"
PROMPT_VERSION = "full-evidence-supported-claim-synthesis-v1"
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

SYNTHESIS_SYSTEM_PROMPT = """You write a concise, evidence-grounded report from a ledger of
previously verified factual claims. Return JSON with a `sections` array. Preserve every supplied
section exactly once and in order. Each section has `sub_narrative` and an `answer` array containing
at most the requested number of items. Each item has `text` and `source_claim_ids`.

Each `text` must be exactly one polished grammatical sentence. Rewrite and merge redundant source
claims, but do not add facts, introductions, conclusions, headings, or transitions unsupported by a
source claim. Use one to three source claim IDs per sentence, all from that section, ordered by
importance. Prefer distinct, information-dense facts that collectively answer the full narrative."""

REPAIR_SYSTEM_PROMPT = """Rewrite one rejected answer sentence so every factual detail is directly
supported by the supplied cited passages. Return JSON with one string field named `text`. Preserve the
supported core, remove or qualify unsupported details, and do not add outside knowledge. The result
must be exactly one concise grammatical sentence with no citation markers or meta-commentary."""


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def validate_submission_entry(entry: Mapping[str, object], *, maximum_words: int = 1024) -> int:
    metadata = entry.get("metadata")
    references = entry.get("references")
    answer = entry.get("answer")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    for key in ("team_id", "narrative_id", "narrative", "run_id", "run_desc"):
        _require_nonempty_string(metadata.get(key), f"metadata.{key}")
    if not isinstance(references, list) or any(
        not isinstance(item, str) or not item.startswith("shard_") for item in references
    ):
        raise ValueError("references must contain exact ClimbMix shard document IDs")
    if len(set(references)) != len(references):
        raise ValueError("references must be unique")
    if not isinstance(answer, list) or not answer:
        raise ValueError("answer must be a nonempty array")
    reference_set = set(references)
    total_words = 0
    for index, sentence in enumerate(answer):
        if not isinstance(sentence, Mapping):
            raise ValueError(f"answer[{index}] must be an object")
        text = _require_nonempty_string(sentence.get("text"), f"answer[{index}].text")
        citations = sentence.get("citations")
        if not isinstance(citations, list) or not 1 <= len(citations) <= 3:
            raise ValueError(f"answer[{index}] must have one to three citations")
        if any(not isinstance(citation, str) or citation not in reference_set for citation in citations):
            raise ValueError(f"answer[{index}] citation is not present in references")
        total_words += len(text.split())
    if total_words > maximum_words:
        raise ValueError(f"answer has {total_words} words; maximum is {maximum_words}")
    return total_words


def synthesize_supported_claims(
    *,
    client: OpenAICompatibleJsonClient,
    source_generation: Mapping[str, object],
    support_rows: Sequence[Mapping[str, object]],
    maximum_sentences_per_section: int,
    target_maximum_words: int,
    max_tokens: int,
    temperature: float,
    validation_attempts: int,
) -> tuple[list[dict[str, object]], dict[str, Mapping[str, object]]]:
    sections = source_generation.get("sections")
    if not isinstance(sections, list):
        raise ValueError("source generation has no sections")
    status_by_id = {str(row["claim_id"]): str(row["status"]) for row in support_rows}
    labels = [str(section["sub_narrative"]) for section in sections if isinstance(section, Mapping)]
    source_by_id: dict[str, Mapping[str, object]] = {}
    claims_by_label: dict[str, list[dict[str, str]]] = {}
    for section in sections:
        if not isinstance(section, Mapping):
            raise ValueError("source section must be an object")
        label = str(section["sub_narrative"])
        claims = section.get("claims")
        if not isinstance(claims, list):
            raise ValueError("source claims must be an array")
        claims_by_label[label] = []
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise ValueError("source claim must be an object")
            claim_id = str(claim["claim_id"])
            if status_by_id.get(claim_id) != "supported":
                continue
            source_by_id[claim_id] = {**dict(claim), "sub_narrative": label}
            claims_by_label[label].append(
                {"claim_id": claim_id, "text": str(claim["text"])}
            )

    payload = {
        "task": "Synthesize a TREC RAG 2026 answer from fully supported source claims.",
        "narrative": source_generation["narrative"],
        "maximum_sentences_per_section": maximum_sentences_per_section,
        "target_maximum_total_words": target_maximum_words,
        "sections": [
            {"sub_narrative": label, "source_claims": claims_by_label[label]}
            for label in labels
        ],
    }
    tokenizer = SpacySentenceTokenizer()

    def validate(value: Mapping[str, object]) -> list[dict[str, object]]:
        raw_sections = value.get("sections")
        if not isinstance(raw_sections, list) or any(
            not isinstance(section, Mapping) for section in raw_sections
        ):
            raise ValueError("sections must be an array of objects")
        by_label: dict[str, Mapping[str, object]] = {}
        for section in raw_sections:
            label = normalize_answer_label(section.get("sub_narrative"), labels)
            if label in by_label:
                raise ValueError(f"duplicate section: {label}")
            by_label[label] = section
        if list(by_label) != labels:
            raise ValueError("all sections must appear exactly once in source order")

        normalized: list[dict[str, object]] = []
        total_words = 0
        for label in labels:
            raw_answer = by_label[label].get("answer")
            if not isinstance(raw_answer, list) or not raw_answer:
                raise ValueError(f"section {label} must have a nonempty answer array")
            if len(raw_answer) > maximum_sentences_per_section:
                raise ValueError(f"section {label} exceeds its sentence budget")
            valid_source_ids = {row["claim_id"] for row in claims_by_label[label]}
            answer: list[dict[str, object]] = []
            for item in raw_answer:
                if not isinstance(item, Mapping):
                    raise ValueError("answer item must be an object")
                text = _require_nonempty_string(item.get("text"), "answer text")
                if len(tokenizer.tokenize(text)) != 1:
                    raise ValueError(f"answer item is not exactly one sentence: {text}")
                source_ids = item.get("source_claim_ids")
                if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 3:
                    raise ValueError("source_claim_ids must contain one to three IDs")
                source_ids = list(dict.fromkeys(map(str, source_ids)))
                if any(source_id not in valid_source_ids for source_id in source_ids):
                    raise ValueError("answer cites an unknown or cross-section source claim")
                total_words += len(text.split())
                answer.append({"text": text, "source_claim_ids": source_ids})
            normalized.append({"sub_narrative": label, "answer": answer})
        if total_words > target_maximum_words:
            raise ValueError(
                f"synthesized answer has {total_words} words; target is {target_maximum_words}"
            )
        return normalized

    synthesized = _complete_validated(
        client,
        stage="trec26_synthesize_supported_claims",
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
        validator=validate,
        validation_attempts=validation_attempts,
    )
    return synthesized, source_by_id


def build_candidate_generation(
    *,
    source_generation: Mapping[str, object],
    synthesized_sections: Sequence[Mapping[str, object]],
    source_by_id: Mapping[str, Mapping[str, object]],
    experiment_id: str,
    run_id: str,
    model: str,
    generation_config: Mapping[str, object],
) -> dict[str, object]:
    output_sections: list[dict[str, object]] = []
    answer_number = 0
    used_source_ids: list[str] = []
    for section in synthesized_sections:
        claims: list[dict[str, object]] = []
        for item in section["answer"]:
            answer_number += 1
            source_ids = list(item["source_claim_ids"])
            supporting_passages: list[dict[str, object]] = []
            document_ids: list[str] = []
            for source_id in source_ids:
                source = source_by_id[source_id]
                raw_passages = source.get("supporting_passages")
                if not isinstance(raw_passages, list) or not raw_passages:
                    raise ValueError(f"source claim {source_id} has no supporting passages")
                if any(not isinstance(passage, Mapping) for passage in raw_passages):
                    raise ValueError("supporting passage must be an object")
                sentence_terms = set(_WORD_RE.findall(str(item["text"]).casefold()))

                def support_score(passage: Mapping[str, object]) -> tuple[float, int]:
                    passage_terms = set(
                        _WORD_RE.findall(str(passage.get("text", "")).casefold())
                    )
                    overlap = len(sentence_terms & passage_terms)
                    coverage = overlap / len(sentence_terms) if sentence_terms else 0.0
                    return coverage, overlap

                passage = max(raw_passages, key=support_score)
                document_id = str(passage["document_id"])
                if not document_id.startswith("shard_"):
                    raise ValueError(f"non-ClimbMix document ID: {document_id}")
                if document_id not in document_ids:
                    document_ids.append(document_id)
                    supporting_passages.append(dict(passage))
                if source_id not in used_source_ids:
                    used_source_ids.append(source_id)
            claims.append(
                {
                    "claim_id": f"S{answer_number:03d}",
                    "text": item["text"],
                    "evidence_claim_ids": source_ids,
                    "document_ids": document_ids,
                    "supporting_passages": supporting_passages,
                }
            )
        output_sections.append(
            {"sub_narrative": section["sub_narrative"], "claims": claims}
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "topic_id": source_generation["topic_id"],
        "narrative": source_generation["narrative"],
        "model": model,
        "temperature": generation_config.get("temperature", 0.0),
        "context_policy": {
            "kind": "synthesis_of_fully_supported_full-evidence_claims",
            "source_experiment_id": source_generation["experiment_id"],
            "nuggets_available_during_generation": False,
        },
        "input_accounting": source_generation["input_accounting"],
        "source_metadata": source_generation["source_metadata"],
        "generation_config": dict(generation_config),
        "prompts": {"synthesis_system": SYNTHESIS_SYSTEM_PROMPT},
        "evidence_ledger": [source_by_id[source_id] for source_id in used_source_ids],
        "sections": output_sections,
    }


def render_markdown(generation: Mapping[str, object]) -> str:
    lines = ["# Topic 213: Korean War", ""]
    for section in generation["sections"]:
        label = str(section["sub_narrative"]).strip().strip('"')
        if label.startswith("New: "):
            label = label[5:]
        lines.extend([f"## {label}", ""])
        for claim in section["claims"]:
            lines.extend(
                [
                    f"{claim['text']} [{'; '.join(claim['document_ids'])}]",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def build_official_entry(
    *, generation: Mapping[str, object], team_id: str, run_desc: str
) -> dict[str, object]:
    references: list[str] = []
    answer: list[dict[str, object]] = []
    for section in generation["sections"]:
        for claim in section["claims"]:
            citations = list(claim["document_ids"])
            for document_id in citations:
                if document_id not in references:
                    references.append(document_id)
            answer.append({"text": claim["text"], "citations": citations})
    entry = {
        "metadata": {
            "team_id": team_id,
            "narrative_id": str(generation["topic_id"]),
            "narrative": generation["narrative"],
            "run_id": generation["run_id"],
            "run_desc": run_desc,
            "generator": generation["model"],
            "source_experiment_id": generation["context_policy"]["source_experiment_id"],
        },
        "references": references,
        "answer": answer,
    }
    validate_submission_entry(entry)
    return entry


def repair_candidate_claim(
    *,
    client: OpenAICompatibleJsonClient,
    claim: Mapping[str, object],
    max_tokens: int,
    validation_attempts: int,
) -> dict[str, object]:
    tokenizer = SpacySentenceTokenizer()
    payload = {
        "rejected_sentence": claim["text"],
        "cited_passages": claim["supporting_passages"],
    }

    def validate(value: Mapping[str, object]) -> str:
        text = _require_nonempty_string(value.get("text"), "repaired text")
        if len(tokenizer.tokenize(text)) != 1:
            raise ValueError("repaired text must be exactly one sentence")
        if len(text.split()) > 60:
            raise ValueError("repaired sentence exceeds 60 words")
        return text

    repaired_text = _complete_validated(
        client,
        stage=f"trec26_repair_{claim['claim_id']}",
        system_prompt=REPAIR_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=max_tokens,
        temperature=0.0,
        validator=validate,
        validation_attempts=validation_attempts,
    )
    return {**dict(claim), "text": repaired_text}


def run_from_config(config_path: Path) -> Path:
    repo_root = find_repo_root(config_path.parent)
    load_repo_env(repo_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    experiment = config["experiment"]
    inputs = config["inputs"]
    generation_config = config["generation"]
    evaluation_config = config["evaluation"]
    if not all(isinstance(item, Mapping) for item in (experiment, inputs, generation_config, evaluation_config)):
        raise ValueError("config sections must be mappings")
    output_dir = repo_root / str(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_generation_path = repo_root / str(inputs["source_generation"])
    source_audit_path = repo_root / str(inputs["source_support_audit"])
    nuggets_path = repo_root / str(inputs["nuggets"])
    source_generation = json.loads(source_generation_path.read_text(encoding="utf-8"))
    support_rows = _read_jsonl(source_audit_path)

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
    synthesized, source_by_id = synthesize_supported_claims(
        client=client,
        source_generation=source_generation,
        support_rows=support_rows,
        maximum_sentences_per_section=int(
            generation_config.get("maximum_sentences_per_section", 3)
        ),
        target_maximum_words=int(generation_config.get("target_maximum_words", 900)),
        max_tokens=int(generation_config.get("max_tokens", 1800)),
        temperature=float(generation_config.get("temperature", 0.0)),
        validation_attempts=int(generation_config.get("validation_attempts", 3)),
    )
    candidate = build_candidate_generation(
        source_generation=source_generation,
        synthesized_sections=synthesized,
        source_by_id=source_by_id,
        experiment_id=str(experiment["id"]),
        run_id=str(experiment["run_id"]),
        model=model,
        generation_config=generation_config,
    )
    full_audit: list[dict[str, object]] = []
    for section in candidate["sections"]:
        full_audit.extend(
            _audit_section(
                client,
                sub_narrative=str(section["sub_narrative"]),
                claims=section["claims"],
                max_tokens=int(generation_config.get("audit_max_tokens", 350)),
                temperature=0.0,
                validation_attempts=int(generation_config.get("validation_attempts", 3)),
            )
        )
    audit_by_id = {str(row["claim_id"]): row for row in full_audit}
    excluded: list[dict[str, object]] = []
    final_audit: list[dict[str, object]] = []
    repair_audit: list[dict[str, object]] = []
    for section in candidate["sections"]:
        kept: list[dict[str, object]] = []
        for claim in section["claims"]:
            audit = audit_by_id[str(claim["claim_id"])]
            if audit["status"] == "supported":
                kept.append(claim)
                final_audit.append(audit)
                continue
            repaired = repair_candidate_claim(
                client=client,
                claim=claim,
                max_tokens=int(generation_config.get("repair_max_tokens", 180)),
                validation_attempts=int(generation_config.get("validation_attempts", 3)),
            )
            repaired_rows = _audit_section(
                client,
                sub_narrative=str(section["sub_narrative"]),
                claims=[repaired],
                max_tokens=int(generation_config.get("audit_max_tokens", 350)),
                temperature=0.0,
                validation_attempts=int(generation_config.get("validation_attempts", 3)),
            )
            repaired_audit = repaired_rows[0]
            repair_audit.append(repaired_audit)
            if repaired_audit["status"] == "supported":
                kept.append(repaired)
                final_audit.append(repaired_audit)
            else:
                excluded.append(
                    {
                        **dict(repaired),
                        "initial_audit": audit,
                        "repair_audit": repaired_audit,
                    }
                )
        section["claims"] = kept
    used_evidence_ids = {
        evidence_id
        for section in candidate["sections"]
        for claim in section["claims"]
        for evidence_id in claim["evidence_claim_ids"]
    }
    candidate["evidence_ledger"] = [
        row for row in candidate["evidence_ledger"] if row["claim_id"] in used_evidence_ids
    ]

    official_entry = build_official_entry(
        generation=candidate,
        team_id=str(experiment["team_id"]),
        run_desc=str(experiment["run_desc"]),
    )
    generation_path = output_dir / "response_generation.json"
    response_path = output_dir / "generated_response.md"
    submission_path = output_dir / "rag_output_trec_rag_2026.jsonl"
    _write_json(generation_path, candidate)
    response_text = render_markdown(candidate)
    response_path.write_text(response_text, encoding="utf-8")
    _write_jsonl(
        output_dir / "generation_support_audit.jsonl",
        [
            *({**row, "audit_pass": "initial"} for row in full_audit),
            *({**row, "audit_pass": "repair"} for row in repair_audit),
        ],
    )
    _write_jsonl(output_dir / "claim_support_audit.jsonl", final_audit)
    _write_jsonl(output_dir / "excluded_sentences.jsonl", excluded)
    _write_jsonl(submission_path, [official_entry])
    frozen_generation_sha256 = _sha256(generation_path)
    frozen_submission_sha256 = _sha256(submission_path)

    nuggets = load_nuggets(
        nuggets_path,
        topic_id=str(source_generation["topic_id"]),
        expected_count=int(evaluation_config.get("expected_nugget_count", 50)),
    )
    comparison = evaluate_nuggets(
        nuggets=nuggets,
        generation=candidate,
        client=client,
        evaluation_config=evaluation_config,
    )
    _write_jsonl(output_dir / "nugget_comparison.jsonl", comparison)
    metrics = compute_evaluation_metrics(
        comparison, final_audit, response_text=response_text
    )
    official_words = validate_submission_entry(official_entry)
    metrics.update(
        {
            "experiment_id": experiment["id"],
            "run_id": experiment["run_id"],
            "topic_id": source_generation["topic_id"],
            "official_submission": {
                "word_count": official_words,
                "sentence_count": len(official_entry["answer"]),
                "reference_count": len(official_entry["references"]),
                "excluded_after_support_audit": len(excluded),
                "maximum_words": 1024,
            },
            "frozen_generation_sha256": frozen_generation_sha256,
            "frozen_submission_sha256": frozen_submission_sha256,
        }
    )
    _write_json(output_dir / "metrics.json", metrics)
    report = f"""# Topic 213: TREC RAG 2026-format regeneration

The original full-evidence response was regenerated from its 42 fully supported claims. Qwen consolidated redundant claims into sentence-level answer objects. Every sentence was re-audited against its selected evidence; sentences not judged fully supported were excluded before the submission artifact was frozen.

## Submission validation

- Sentences: {len(official_entry['answer'])}
- Answer words: {official_words} / 1024
- ClimbMix references: {len(official_entry['references'])}
- Maximum citations per sentence: 3
- Sentences excluded after support audit: {len(excluded)}
- Citation format: direct `shard_*` document IDs

## Nugget evaluation

- Strict coverage: {metrics['nuggets']['all']['strict_coverage']:.3f}
- Partial-credit coverage: {metrics['nuggets']['all']['partial_credit_coverage']:.3f}
- Vital strict coverage: {metrics['nuggets']['vital']['strict_coverage']:.3f}
- Citation coverage: {metrics['answer_claims']['citation_coverage']:.3f}
- Unsupported submitted sentences: {metrics['answer_claims']['unsupported_claim_count']}

The organizer nuggets remained unavailable until the generation, support audit, and official JSONL submission were frozen.
"""
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    _write_json(output_dir / "config.resolved.json", json.loads(json.dumps(config)))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_from_config(args.config.resolve()))


if __name__ == "__main__":
    main()
