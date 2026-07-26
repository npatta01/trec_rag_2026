"""Apply Ragnarok top-20 answer generation and citation postprocessing to Issue #19."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import yaml

from trec_rag.repo_env import find_repo_root, load_repo_env
from trec_rag.topic213_response_experiment import (
    OpenAICompatibleJsonClient,
    PassageCorpus,
    PassageUnit,
    _audit_section,
    _clean_heading,
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
from trec_rag.topic213_utokyo_experiment import normalize_answer_label


SCHEMA_VERSION = "topic213-ragnarok-experiment-v1"
PROMPT_VERSION = "ragnarok-v4-issue19-outline-v1"
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_TRAILING_META_RE = re.compile(r"\n(?:Note|References):.*", re.IGNORECASE | re.DOTALL)

RAGNAROK_SYSTEM_PROMPT = (
    "This is a chat between a user and an artificial intelligence assistant. "
    "The assistant gives helpful and detailed answers to the user's question based on "
    "the context references. The assistant should also indicate when the answer cannot "
    "be found in the context references."
)

RAGNAROK_INSTRUCTION = """Provide a concise, information-dense answer to the question. Your response
must not exceed 380 words. Cite supporting context documents inline using IEEE-style square brackets.
Include 1-3 citations per sentence, ordered by decreasing importance, and ensure every factual sentence
has at least one citation. Use multiple sources when useful, acknowledge contradictions, express
uncertainty when appropriate, and avoid unsupported claims or meta-commentary about the references.

For this experiment, preserve the supplied ten sub-narratives as Markdown level-two headings in the
exact supplied order. Under each heading, write only cited prose sentences supported by the supplied
documents. A heading may have no prose when the selected evidence does not support that facet. Do not
write an introduction, conclusion, bullets, a references list, or text outside those ten sections."""


class TextAndJsonCompletionClient(Protocol):
    model: str

    def complete_text(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str: ...

    def complete_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: Mapping[str, object],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]: ...


class SpacySentenceTokenizer:
    """Sentence splitter backed by SpaCy, with a no-download sentencizer fallback."""

    def __init__(self, model: str | None = None) -> None:
        try:
            import spacy
        except ModuleNotFoundError as exc:
            raise RuntimeError("SpaCy is required; run `uv sync --group utokyo`") from exc

        self.requested_model = model
        self.loaded_model = model or "spacy.blank(en)+sentencizer"
        if model:
            try:
                self.nlp = spacy.load(model)
            except OSError:
                self.nlp = spacy.blank("en")
                self.nlp.add_pipe("sentencizer")
                self.loaded_model = "spacy.blank(en)+sentencizer"
        else:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")

    def tokenize(self, text: str) -> list[str]:
        normalized = " ".join(text.replace("\n", " ").split())
        if not normalized:
            return []
        return [sent.text.strip() for sent in self.nlp(normalized).sents if sent.text.strip()]


def build_ragnarok_user_prompt(
    *, narrative: str, labels: Sequence[str], passages: Sequence[PassageUnit]
) -> str:
    references = "\n".join(
        f"[{rank}] {' '.join(passage.text.replace(chr(10), ' ').split())}"
        for rank, passage in enumerate(passages, 1)
    )
    outline = "\n".join(f"## {label}" for label in labels)
    return (
        f"Instruction: {RAGNAROK_INSTRUCTION}\n\n"
        f"Documents:\n{references}\n\n"
        f"Query: {narrative}\n\n"
        f"Required outline:\n{outline}\n\n"
        f"Instruction: {RAGNAROK_INSTRUCTION}\n\nAnswer:"
    )


def _parse_sentence_citations(sentence: str, reference_count: int) -> dict[str, object]:
    citation_indexes: list[int] = []
    for match in _CITATION_RE.finditer(sentence):
        citation_indexes.extend(int(value.strip()) - 1 for value in match.group(1).split(","))
    cleaned = _CITATION_RE.sub("", sentence)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if re.search(r"\[\s*\d", cleaned):
        raise ValueError(f"malformed citation in sentence: {sentence}")
    citation_indexes = list(dict.fromkeys(citation_indexes))
    if not citation_indexes:
        raise ValueError(f"uncited sentence: {cleaned}")
    if len(citation_indexes) > 3:
        raise ValueError(f"sentence has more than three citations: {cleaned}")
    if any(index < 0 or index >= reference_count for index in citation_indexes):
        raise ValueError(f"citation outside the top-{reference_count} references: {sentence}")
    if not cleaned:
        raise ValueError("citation-only sentence is not an answer")
    return {"text": cleaned, "citations": citation_indexes}


def postprocess_ragnarok_response(
    raw_response: str,
    *,
    labels: Sequence[str],
    reference_count: int,
    tokenizer: SpacySentenceTokenizer,
    maximum_words: int = 380,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split headed prose with SpaCy and convert one-based citations to reference indexes."""
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = _TRAILING_META_RE.sub("", text)
    raw_sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        heading = _HEADING_RE.match(line.strip())
        if heading:
            if current_heading is not None:
                raw_sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = heading.group(1).strip()
            current_lines = []
        elif line.strip():
            if current_heading is None:
                raise ValueError("response contains text before the first required heading")
            current_lines.append(line.strip())
    if current_heading is not None:
        raw_sections.append((current_heading, "\n".join(current_lines).strip()))

    by_label: dict[str, str] = {}
    for heading, body in raw_sections:
        label = normalize_answer_label(heading, labels)
        if label in by_label:
            raise ValueError(f"duplicate response section: {label}")
        by_label[label] = body
    if list(by_label) != list(labels):
        raise ValueError("response must contain every sub-narrative heading in the supplied order")

    answer: list[dict[str, object]] = []
    mapped_sections: list[dict[str, object]] = []
    for label in labels:
        section_sentences: list[dict[str, object]] = []
        for sentence in tokenizer.tokenize(by_label[label]):
            parsed = _parse_sentence_citations(sentence, reference_count)
            answer.append(parsed)
            section_sentences.append(parsed)
        mapped_sections.append(
            {"sub_narrative": label, "sentences": section_sentences}
        )
    response_length = sum(len(str(sentence["text"]).split()) for sentence in answer)
    if response_length > maximum_words:
        raise ValueError(
            f"response has {response_length} words after citation removal; limit is {maximum_words}"
        )
    if not answer:
        raise ValueError("response contains no cited answer sentences")
    return answer, mapped_sections


def _complete_validated_text(
    client: TextAndJsonCompletionClient,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    validation_attempts: int,
    validator,
):
    last_error: Exception | None = None
    current_prompt = user_prompt
    for attempt in range(1, validation_attempts + 1):
        raw = client.complete_text(
            stage="ragnarok_generate_top20" if attempt == 1 else f"ragnarok_generate_top20_repair_{attempt}",
            system_prompt=system_prompt,
            user_prompt=current_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            return raw, validator(raw)
        except ValueError as exc:
            last_error = exc
            current_prompt = (
                f"{user_prompt}\n\nYour prior response was invalid: {exc}. Regenerate the full answer "
                "more concisely and obey every formatting and citation constraint."
            )
        raise RuntimeError("Ragnarok response remained invalid after repair attempts") from last_error


def generate_ragnarok_top20_response(
    *,
    client: TextAndJsonCompletionClient,
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
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    passage_lookup = {passage.passage_id: passage for passage in corpus.passages}
    if len(selected_passage_ids) != int(generation_config.get("evidence_depth", 20)):
        raise ValueError("retrieval artifact must provide the configured top-20 evidence depth")
    if len(set(selected_passage_ids)) != len(selected_passage_ids):
        raise ValueError("selected passage IDs must be unique")
    if any(passage_id not in passage_lookup for passage_id in selected_passage_ids):
        raise ValueError("retrieval artifact selects a passage outside the release corpus")
    selected = [passage_lookup[passage_id] for passage_id in selected_passage_ids]
    tokenizer = SpacySentenceTokenizer(
        str(generation_config["spacy_model"]) if generation_config.get("spacy_model") else None
    )
    user_prompt = build_ragnarok_user_prompt(
        narrative=narrative, labels=corpus.sub_narratives, passages=selected
    )

    def validate(raw: str):
        return postprocess_ragnarok_response(
            raw,
            labels=corpus.sub_narratives,
            reference_count=len(selected),
            tokenizer=tokenizer,
            maximum_words=int(generation_config.get("maximum_words", 380)),
        )

    raw_response, (answer, mapped_sections) = _complete_validated_text(
        client,
        system_prompt=RAGNAROK_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=int(generation_config.get("answer_max_tokens", 1600)),
        temperature=float(generation_config.get("temperature", 0.0)),
        validation_attempts=int(generation_config.get("validation_attempts", 3)),
        validator=validate,
    )

    evidence_ids: dict[tuple[str, str], str] = {}
    evidence_ledger: list[dict[str, object]] = []
    sections: list[dict[str, object]] = []
    sentence_rows: list[dict[str, object]] = []
    answer_number = 0
    for mapped_section in mapped_sections:
        label = str(mapped_section["sub_narrative"])
        claims: list[dict[str, object]] = []
        for sentence in mapped_section["sentences"]:
            answer_number += 1
            indexes = list(sentence["citations"])
            cited_passages = [selected[index] for index in indexes]
            linked_evidence_ids: list[str] = []
            for passage in cited_passages:
                key = (label, passage.passage_id)
                if key not in evidence_ids:
                    evidence_id = f"RGE{len(evidence_ledger) + 1:03d}"
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
            supporting_passages = [
                {
                    "passage_id": passage.passage_id,
                    "document_id": passage.document_id,
                    "text": passage.text,
                }
                for passage in cited_passages
            ]
            document_ids = list(
                dict.fromkeys(passage.document_id for passage in cited_passages)
            )
            claim = {
                "claim_id": f"A{answer_number:03d}",
                "text": sentence["text"],
                "evidence_claim_ids": linked_evidence_ids,
                "reference_indexes": indexes,
                "passage_ids": [passage.passage_id for passage in cited_passages],
                "document_ids": document_ids,
                "supporting_passages": supporting_passages,
            }
            claims.append(claim)
            sentence_rows.append({"sub_narrative": label, **claim})
        sections.append({"sub_narrative": label, "claims": claims})

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

    references = list(selected_passage_ids)
    response_length = sum(len(str(sentence["text"]).split()) for sentence in answer)
    ragnarok_record = {
        "run_id": run_id,
        "topic_id": topic_id,
        "topic": narrative,
        "references": references,
        "response_length": response_length,
        "answer": answer,
        "metadata": {
            "team_id": "issue19-local",
            "run_id": run_id,
            "narrative_id": topic_id,
            "narrative": narrative,
            "type": "automatic",
            "prompt": PROMPT_VERSION,
        },
    }
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
            "kind": "utokyo_reranked_top20_ragnarok_v4",
            "retrieval_corpus": "all released passages",
            "generation_evidence": "frozen final reranked top 20 passages",
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
            "ragnarok_system": RAGNAROK_SYSTEM_PROMPT,
            "ragnarok_instruction": RAGNAROK_INSTRUCTION,
            "ragnarok_user": user_prompt,
        },
        "postprocessing": {
            "sentence_tokenizer": tokenizer.loaded_model,
            "citation_syntax": "one-based IEEE inline markers",
            "output_citation_semantics": "zero-based indexes into references",
            "raw_response": raw_response,
        },
        "evidence_ledger": evidence_ledger,
        "sections": sections,
    }
    return generation, support_audit, {
        "ragnarok_record": ragnarok_record,
        "raw_response": raw_response,
        "sentence_rows": sentence_rows,
    }


def _resolve(repo_root: Path, value: object, label: str) -> Path:
    path = Path(_require_nonempty_string(value, label))
    return path if path.is_absolute() else repo_root / path


def _render_report(
    *, generation: Mapping[str, object], metrics: Mapping[str, object]
) -> str:
    base = render_evaluation_report(generation=generation, metrics=metrics)
    base = base.replace(
        "# Topic 213 response-generation evaluation",
        "# Topic 213 Ragnarok generation evaluation",
        1,
    )
    base = base.replace(
        "The full released passage packet was consumed during generation. Two eligible documents are absent because their original provider calls failed before inference, and this experiment does not fabricate or replace their evidence.",
        "The upstream UTokyo retrieval run scored all 1,478 passages from the 171 available documents; this generation stage used its frozen top 20. Two of 173 eligible documents remain absent because their original provider calls failed before inference.",
    )
    lines = [
        base.rstrip(),
        "",
        "## Steps 8 and 9",
        "",
        "The frozen UTokyo reranker top 20 was sent as an ordered reference list to local Qwen through LiteLLM. The model generated one IEEE-cited response under the ten Issue #19 sub-narrative headings. SpaCy then split each section into sentences, removed inline citation markers, converted one-based markers to zero-based indexes into the ordered references, and resolved those references to release passage and document IDs.",
        "",
        f"- Ragnarok answer words (submission schema): {metrics['ragnarok']['response_words']}",
        f"- Rendered Markdown words (headings and document IDs included): {metrics['response']['word_count']}",
        f"- Sentence claims: {metrics['answer_claims']['total']}",
        f"- Top-20 references cited: {metrics['ragnarok']['cited_references']} / 20",
        f"- Sub-narratives with at least one sentence: {metrics['ragnarok']['nonempty_sections']} / 10",
        f"- SpaCy pipeline: {generation['postprocessing']['sentence_tokenizer']}",
        "",
    ]
    return "\n".join(lines)


def run_from_config(config_path: Path) -> Path:
    repo_root = find_repo_root(config_path.parent)
    load_repo_env(repo_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("experiment config must be a YAML mapping")
    for key in ("experiment", "inputs", "generation", "evaluation"):
        if not isinstance(config.get(key), Mapping):
            raise ValueError(f"config requires a {key} mapping")
    experiment = config["experiment"]
    inputs = config["inputs"]
    generation_config = config["generation"]
    evaluation_config = config["evaluation"]
    assert all(
        isinstance(item, Mapping)
        for item in (experiment, inputs, generation_config, evaluation_config)
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
        "retrieval_run": _resolve(
            repo_root, inputs.get("retrieval_run"), "inputs.retrieval_run"
        ),
    }
    for key, path in paths.items():
        expected = inputs.get(f"{key}_sha256")
        if expected and str(expected).casefold() != _sha256(path).casefold():
            raise ValueError(f"{key}_sha256 does not match {path}")

    corpus = load_passage_corpus(paths["passages"])
    accounting = validate_release_accounting(
        corpus, load_release_manifest(paths["release_manifest"])
    )
    narrative = load_topic_narrative(paths["topics"], topic_id=topic_id)
    retrieval_run = json.loads(paths["retrieval_run"].read_text(encoding="utf-8"))
    if not isinstance(retrieval_run, Mapping):
        raise ValueError("retrieval run must be a JSON object")
    selected_ids = retrieval_run.get("selected_passage_ids")
    if not isinstance(selected_ids, list) or any(not isinstance(item, str) for item in selected_ids):
        raise ValueError("retrieval run must contain selected_passage_ids")

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
    source_metadata = {
        f"{name}_path": str(path.relative_to(repo_root)) for name, path in paths.items()
    }
    source_metadata.update(
        {f"{name}_sha256": _sha256(path) for name, path in paths.items()}
    )
    source_metadata["passages_release_url"] = inputs.get("passages_release_url")
    source_metadata["trec_rag_data_revision"] = inputs.get("trec_rag_data_revision")

    generation, support_audit, artifacts = generate_ragnarok_top20_response(
        client=client,
        topic_id=topic_id,
        narrative=narrative,
        corpus=corpus,
        selected_passage_ids=selected_ids,
        release_accounting=accounting,
        retrieval_record={
            "retrieval_run_path": str(paths["retrieval_run"].relative_to(repo_root)),
            "retrieval_run_sha256": _sha256(paths["retrieval_run"]),
            "selected_passage_ids": selected_ids,
        },
        generation_config=generation_config,
        experiment_id=experiment_id,
        run_id=run_id,
        source_metadata=source_metadata,
    )
    generation_path = output_dir / "response_generation.json"
    response_path = output_dir / "generated_response.md"
    support_path = output_dir / "claim_support_audit.jsonl"
    _write_json(generation_path, generation)
    response_text = render_generated_response(generation)
    response_path.write_text(response_text, encoding="utf-8")
    _write_jsonl(support_path, support_audit)
    _write_jsonl(output_dir / "ragnarok_response.jsonl", [artifacts["ragnarok_record"]])
    (output_dir / "ragnarok_raw_response.txt").write_text(
        str(artifacts["raw_response"]).rstrip() + "\n", encoding="utf-8"
    )
    _write_jsonl(output_dir / "sentence_citations.jsonl", artifacts["sentence_rows"])
    frozen_generation_sha256 = _sha256(generation_path)
    frozen_response_sha256 = _sha256(response_path)

    # Organizer nuggets are first read after all generation artifacts are frozen.
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
    answer = artifacts["ragnarok_record"]["answer"]
    cited_indexes = {
        index for sentence in answer for index in sentence["citations"]
    }
    nonempty_sections = sum(bool(section["claims"]) for section in generation["sections"])
    metrics.update(
        {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "topic_id": topic_id,
            "frozen_generation_sha256": frozen_generation_sha256,
            "frozen_response_sha256": frozen_response_sha256,
            "ragnarok": {
                "references": len(selected_ids),
                "cited_references": len(cited_indexes),
                "response_words": artifacts["ragnarok_record"]["response_length"],
                "nonempty_sections": nonempty_sections,
                "sentence_tokenizer": generation["postprocessing"]["sentence_tokenizer"],
                "citation_index_base": 0,
                "selected_released_sub_narratives": dict(
                    Counter(
                        passage.sub_narrative
                        for passage in corpus.passages
                        if passage.passage_id in set(selected_ids)
                    )
                ),
            },
        }
    )
    _write_json(output_dir / "metrics.json", metrics)
    (output_dir / "evaluation_report.md").write_text(
        _render_report(generation=generation, metrics=metrics), encoding="utf-8"
    )
    _write_json(output_dir / "config.resolved.json", json.loads(json.dumps(config)))
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    print(run_from_config(args.config.resolve()))


if __name__ == "__main__":
    main()
