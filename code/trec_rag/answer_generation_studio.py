"""Run and serve the selected competition answer-generation approach."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from trec_rag.topic213_controlled_generator_benchmark import (
    GENERATION_SYSTEM_PROMPT,
    GeneratorClient,
    SingleCandidateJsonClient,
    _make_judge_client,
    build_candidate_generation,
    build_generation_payload,
    build_official_entry,
    build_response_schema,
    filter_supported_generation,
    semantic_request_sha256,
    validate_generation_response,
    validate_official_entry,
)
from trec_rag.topic213_ragnarok_experiment import SpacySentenceTokenizer
from trec_rag.topic213_response_experiment import (
    JsonCompletionClient,
    _audit_section,
    _require_nonempty_string,
    _sha256,
    _write_json,
    _write_jsonl,
)


STUDIO_SCHEMA_VERSION = "trec-rag-answer-generation-studio-v1"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class GenerationRunResult:
    output_dir: Path
    summary: dict[str, object]
    official_entry: dict[str, object]
    final_generation: dict[str, object]


def _emit(
    progress: ProgressCallback | None,
    *,
    stage: str,
    message: str,
    completed: int = 0,
    total: int = 1,
) -> None:
    if progress is not None:
        progress(
            {
                "stage": stage,
                "message": message,
                "completed": completed,
                "total": total,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )


def validate_run_id(value: str) -> str:
    run_id = value.strip()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run ID must start with a letter or digit and contain only letters, digits, ., _, or -"
        )
    return run_id


def validate_frozen_evidence_for_generation(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate a nugget-blind frozen ledger without weakening its source contract."""

    frozen = copy.deepcopy(dict(value))
    if frozen.get("organizer_nuggets_available") is not False:
        raise ValueError("generation evidence must be explicitly nugget-blind")
    _require_nonempty_string(frozen.get("topic_id"), "topic_id")
    _require_nonempty_string(frozen.get("narrative"), "narrative")
    _require_nonempty_string(frozen.get("source_experiment_id"), "source_experiment_id")

    facets = frozen.get("facets")
    if not isinstance(facets, list) or not facets:
        raise ValueError("frozen evidence must contain at least one facet")
    expected_position = 0
    seen_claim_ids: set[str] = set()
    source_word_count = 0
    for facet_index, facet in enumerate(facets, 1):
        if not isinstance(facet, Mapping):
            raise ValueError(f"facet {facet_index} must be an object")
        label = _require_nonempty_string(facet.get("sub_narrative"), "sub_narrative")
        claims = facet.get("source_claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError(f"facet {facet_index} must contain source claims")
        if int(facet.get("sentence_quota", -1)) != len(claims):
            raise ValueError(f"facet {facet_index} sentence quota does not match its claims")
        if int(facet.get("facet_number", -1)) != facet_index:
            raise ValueError("facet numbers must be consecutive and ordered")
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise ValueError(f"facet {facet_index} contains a non-object claim")
            expected_position += 1
            if int(claim.get("position", -1)) != expected_position:
                raise ValueError("claim positions must be consecutive and ordered")
            claim_id = _require_nonempty_string(claim.get("claim_id"), "claim_id")
            if claim_id in seen_claim_ids:
                raise ValueError(f"duplicate source claim ID: {claim_id}")
            seen_claim_ids.add(claim_id)
            if str(claim.get("sub_narrative")) != label:
                raise ValueError(f"claim {claim_id} is assigned to the wrong facet")
            text = _require_nonempty_string(claim.get("text"), "source claim text")
            source_word_count += len(text.split())
            document_ids = claim.get("document_ids")
            passages = claim.get("selected_citation_passages")
            if (
                not isinstance(document_ids, list)
                or not 1 <= len(document_ids) <= 3
                or len(set(map(str, document_ids))) != len(document_ids)
            ):
                raise ValueError(f"claim {claim_id} must have one to three unique citations")
            if any(not str(document_id).startswith("shard_") for document_id in document_ids):
                raise ValueError(f"claim {claim_id} contains a non-ClimbMix citation")
            if not isinstance(passages, list) or len(passages) != len(document_ids):
                raise ValueError(f"claim {claim_id} citation passages do not match document IDs")
            passage_ids = []
            for passage in passages:
                if not isinstance(passage, Mapping):
                    raise ValueError(f"claim {claim_id} contains a non-object passage")
                passage_ids.append(
                    _require_nonempty_string(passage.get("document_id"), "passage document_id")
                )
                _require_nonempty_string(passage.get("text"), "passage text")
            if passage_ids != list(map(str, document_ids)):
                raise ValueError(f"claim {claim_id} passage order does not match citations")

    declared_count = int(frozen.get("supported_source_claim_count", -1))
    exact_count = int(frozen.get("exact_total_sentence_count", -1))
    if declared_count != expected_position or exact_count != expected_position:
        raise ValueError("frozen claim counts do not match the evidence ledger")
    if int(frozen.get("source_claim_word_count", -1)) != source_word_count:
        raise ValueError("source claim word count does not match the evidence ledger")
    word_band = frozen.get("word_band")
    if not isinstance(word_band, Mapping):
        raise ValueError("frozen evidence is missing a word band")
    minimum = int(word_band.get("minimum", -1))
    maximum = int(word_band.get("maximum", -1))
    if minimum < 1 or minimum > maximum or maximum > 1024:
        raise ValueError("frozen evidence has an invalid organizer word band")
    return frozen


def _artifact_row(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _render_markdown(generation: Mapping[str, object]) -> str:
    lines = [f"# Topic {generation['topic_id']}", "", str(generation["narrative"]), ""]
    for section in generation["sections"]:
        lines.extend([f"## {str(section['sub_narrative']).strip().strip(chr(34))}", ""])
        claims = section["claims"]
        if not claims:
            lines.extend(["No sentence survived the support audit.", ""])
        for claim in claims:
            lines.extend([f"{claim['text']} [{'; '.join(claim['document_ids'])}]", ""])
    return "\n".join(lines).rstrip() + "\n"


def _safe_config(config: Mapping[str, object]) -> dict[str, object]:
    generation = config["generation"]
    audit = config["audit"]
    return {
        "schema_version": STUDIO_SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "generation": {
            "key": generation["key"],
            "model": generation["model"],
            "model_identity": generation["model_identity"],
            "temperature": float(generation.get("temperature", 0.0)),
            "max_tokens": int(generation.get("max_tokens", 3500)),
        },
        "audit": {
            "model_identity": audit.get("model_identity", "qwen-local"),
            "temperature": float(audit.get("temperature", 0.0)),
            "max_tokens": int(audit.get("max_tokens", 350)),
            "validation_attempts": int(audit.get("validation_attempts", 3)),
        },
        "organizer_nuggets_available": False,
    }


def _generator_client(
    *, output_dir: Path, config: Mapping[str, object]
) -> SingleCandidateJsonClient:
    env_name = str(config.get("api_key_env", "OPENROUTER_API_KEY"))
    api_key = os.environ.get(env_name, str(config.get("api_key", ""))).strip()
    if not api_key:
        raise ValueError(f"{env_name} is required for the selected generator")
    api_base = os.environ.get(
        str(config.get("api_base_env", "")), str(config.get("api_base", ""))
    ).strip()
    if not api_base:
        raise ValueError("generation API base is required")
    return SingleCandidateJsonClient(
        api_base=api_base,
        model=str(config["model"]),
        api_key=api_key,
        checkpoint_path=output_dir / "generation_checkpoint.json",
        call_log_path=output_dir / "generation_calls.jsonl",
        timeout_seconds=float(config.get("timeout_seconds", 360)),
        transport_max_attempts=int(config.get("transport_max_attempts", 3)),
        request_overrides=config.get("request_overrides", {}),
    )


def run_best_answer_generation(
    *,
    frozen: Mapping[str, object],
    output_dir: Path,
    run_id: str,
    team_id: str,
    run_desc: str,
    config: Mapping[str, object],
    generator_client: GeneratorClient | None = None,
    judge_client: JsonCompletionClient | None = None,
    progress: ProgressCallback | None = None,
) -> GenerationRunResult:
    """Generate, audit, validate, and freeze one official organizer response."""

    run_id = validate_run_id(run_id)
    team_id = _require_nonempty_string(team_id, "team_id")
    run_desc = _require_nonempty_string(run_desc, "run_desc")
    validated_frozen = validate_frozen_evidence_for_generation(frozen)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, stage="preflight", message="Evidence contract validated", completed=1)

    generation_config = config["generation"]
    audit_config = config["audit"]
    payload = build_generation_payload(validated_frozen)
    response_schema = build_response_schema(
        int(validated_frozen["exact_total_sentence_count"])
    )
    max_tokens = int(generation_config.get("max_tokens", 3500))
    temperature = float(generation_config.get("temperature", 0.0))
    semantic_hash = semantic_request_sha256(
        system_prompt=GENERATION_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    _write_json(output_dir / "run_config.json", _safe_config(config))
    _write_json(output_dir / "frozen_evidence_ledger.json", validated_frozen)
    _write_json(output_dir / "frozen_generation_payload.json", payload)
    _write_json(output_dir / "response_schema.json", response_schema)
    (output_dir / "generation_system_prompt.txt").write_text(
        GENERATION_SYSTEM_PROMPT + "\n", encoding="utf-8"
    )

    _emit(progress, stage="generation", message="Generating one claim-preserving candidate")
    active_generator = generator_client or _generator_client(
        output_dir=output_dir, config=generation_config
    )
    completion = active_generator.complete_once(
        system_prompt=GENERATION_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if completion.receipt.get("semantic_request_sha256") != semantic_hash:
        raise ValueError("generator receipt does not match the frozen semantic request")
    tokenizer = SpacySentenceTokenizer()
    normalized = validate_generation_response(
        completion.parsed, frozen=validated_frozen, tokenizer=tokenizer
    )
    _emit(progress, stage="generation", message="Candidate contract validated", completed=1)

    source_generation = {
        "experiment_id": validated_frozen["source_experiment_id"],
        "topic_id": validated_frozen["topic_id"],
        "narrative": validated_frozen["narrative"],
    }
    candidate = build_candidate_generation(
        source_generation=source_generation,
        frozen=validated_frozen,
        normalized_sentences=normalized,
        model_key=str(generation_config["key"]),
        model_identity=str(generation_config["model_identity"]),
        receipt=completion.receipt,
        experiment_id=str(config["experiment_id"]),
        run_id=run_id,
    )

    active_judge = judge_client or _make_judge_client(
        output_dir=output_dir, config=audit_config, purpose="support_audit"
    )
    audit_rows: list[dict[str, object]] = []
    sections = candidate["sections"]
    for section_index, section in enumerate(sections, 1):
        audit_rows.extend(
            _audit_section(
                active_judge,
                sub_narrative=str(section["sub_narrative"]),
                claims=section["claims"],
                max_tokens=int(audit_config.get("max_tokens", 350)),
                temperature=float(audit_config.get("temperature", 0.0)),
                validation_attempts=int(audit_config.get("validation_attempts", 3)),
            )
        )
        _emit(
            progress,
            stage="support_audit",
            message=f"Audited facet {section_index} of {len(sections)}",
            completed=section_index,
            total=len(sections),
        )

    final, kept_audits, excluded = filter_supported_generation(candidate, audit_rows)
    official = build_official_entry(
        generation=final,
        team_id=team_id,
        run_desc=f"{run_desc} Generator: {generation_config['model_identity']}.",
    )
    official_words = validate_official_entry(official, tokenizer=tokenizer)
    markdown = _render_markdown(final)
    candidate_words = sum(
        len(str(claim["text"]).split())
        for section in candidate["sections"]
        for claim in section["claims"]
    )

    _write_json(output_dir / "raw_generation.json", completion.parsed)
    (output_dir / "raw_generation.txt").write_text(
        completion.raw_content, encoding="utf-8"
    )
    _write_json(output_dir / "generation_receipt.json", completion.receipt)
    _write_json(output_dir / "response_generation.candidate.json", candidate)
    _write_json(output_dir / "response_generation.json", final)
    _write_jsonl(output_dir / "generation_support_audit.jsonl", audit_rows)
    _write_jsonl(output_dir / "claim_support_audit.jsonl", kept_audits)
    _write_jsonl(output_dir / "excluded_sentences.jsonl", excluded)
    _write_jsonl(output_dir / "rag_output_trec_rag_2026.jsonl", [official])
    (output_dir / "generated_response.md").write_text(markdown, encoding="utf-8")
    lineage = [
        {
            "sentence_claim_id": claim["claim_id"],
            "source_claim_id": claim["evidence_claim_ids"][0],
            "sub_narrative": section["sub_narrative"],
            "document_ids": claim["document_ids"],
            "submitted": any(
                claim["claim_id"] == kept["claim_id"] for kept in kept_audits
            ),
        }
        for section in candidate["sections"]
        for claim in section["claims"]
    ]
    _write_jsonl(output_dir / "lineage.jsonl", lineage)

    frozen_names = [
        "run_config.json",
        "frozen_evidence_ledger.json",
        "frozen_generation_payload.json",
        "response_schema.json",
        "generation_system_prompt.txt",
        "raw_generation.json",
        "raw_generation.txt",
        "generation_receipt.json",
        "response_generation.candidate.json",
        "response_generation.json",
        "generation_support_audit.jsonl",
        "claim_support_audit.jsonl",
        "excluded_sentences.jsonl",
        "rag_output_trec_rag_2026.jsonl",
        "generated_response.md",
        "lineage.jsonl",
    ]
    freeze = {
        "schema_version": "competition-answer-generation-freeze-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "organizer_nuggets_read": False,
        "semantic_request_sha256": semantic_hash,
        "artifacts": [_artifact_row(output_dir, name) for name in frozen_names],
    }
    _write_json(output_dir / "generation_freeze.json", freeze)
    receipt_usage = completion.receipt.get("usage", {})
    cost = (
        float(receipt_usage.get("cost", 0.0))
        if isinstance(receipt_usage, Mapping)
        and isinstance(receipt_usage.get("cost", 0.0), (int, float))
        else 0.0
    )
    summary = {
        "schema_version": STUDIO_SCHEMA_VERSION,
        "status": "valid",
        "official_format_valid": True,
        "organizer_nuggets_read": False,
        "topic_id": validated_frozen["topic_id"],
        "narrative": validated_frozen["narrative"],
        "run_id": run_id,
        "team_id": team_id,
        "generator": generation_config["model_identity"],
        "support_auditor": audit_config.get("model_identity", "qwen-local"),
        "semantic_request_sha256": semantic_hash,
        "generation_freeze_sha256": _sha256(output_dir / "generation_freeze.json"),
        "candidate_sentence_count": len(normalized),
        "submitted_sentence_count": len(kept_audits),
        "excluded_sentence_count": len(excluded),
        "candidate_word_count": candidate_words,
        "submitted_word_count": official_words,
        "reference_count": len(official["references"]),
        "citation_coverage": (
            sum(bool(item["citations"]) for item in official["answer"])
            / len(official["answer"])
            if official["answer"]
            else 0.0
        ),
        "maximum_words": 1024,
        "estimated_generation_cost": cost,
        "submission_file": "rag_output_trec_rag_2026.jsonl",
    }
    _write_json(output_dir / "summary.json", summary)
    manifest_names = [*frozen_names, "generation_freeze.json", "summary.json"]
    manifest = {
        "schema_version": "competition-answer-generation-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "topic_id": validated_frozen["topic_id"],
        "organizer_nuggets_read": False,
        "model": generation_config["model_identity"],
        "artifacts": [_artifact_row(output_dir, name) for name in manifest_names],
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    _emit(progress, stage="complete", message="Organizer submission frozen", completed=1)
    return GenerationRunResult(output_dir, summary, official, final)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("studio config must be a mapping")
    from trec_rag.repo_env import find_repo_root, load_repo_env

    repo_root = find_repo_root(config_path.parent)
    load_repo_env(repo_root)
    app_config = config["app"]
    evidence_path = args.evidence or repo_root / str(app_config["default_evidence"])
    frozen = json.loads(evidence_path.read_text(encoding="utf-8"))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("rag26-%Y%m%d-%H%M%S")
    output_dir = args.output_dir or repo_root / str(app_config["run_root"]) / run_id
    result = run_best_answer_generation(
        frozen=frozen,
        output_dir=output_dir,
        run_id=run_id,
        team_id=str(config["submission"]["team_id"]),
        run_desc=str(config["submission"]["run_desc"]),
        config=config,
        progress=lambda event: print(f"[{event['stage']}] {event['message']}", flush=True),
    )
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
