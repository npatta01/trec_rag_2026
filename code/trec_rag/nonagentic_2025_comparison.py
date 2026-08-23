"""Compare one non-agentic 2025 handoff topic with organizer nuggets."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any

import yaml


WORD_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "for",
    "from", "how", "in", "include", "including", "is", "it", "of", "on",
    "or", "the", "this", "to", "was", "were", "what", "with",
}
GENERIC = {"korean", "korea", "war", "us", "united", "states"}
CONCEPT_ALIASES = {
    "origin": ("origin", "outbreak", "trigger", "begin", "invasion", "divis", "hostilit"),
    "conclusion": ("conclud", "end", "armistice", "dmz", "demilitar", "negotiat"),
    "motivation": ("motivat", "involv", "containment", "cold", "strategy", "reason"),
    "politics": ("politic", "impact", "domestic", "foreign", "policy"),
    "mistakes": ("mistake", "error", "strategic", "operational", "failure", "misstep"),
    "presidents": ("president", "truman", "eisenhower", "perception", "view"),
    "peninsula": ("divid", "peninsula", "38th"),
    "un": ("un", "nations", "resolution"),
    "china": ("china", "chinese", "yal"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be an object")
        rows.append(value)
    return rows


def _stem(token: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokens(text: str) -> set[str]:
    return {
        _stem(token)
        for token in WORD_RE.findall(text.lower())
        if token not in STOPWORDS and token not in GENERIC
    }


def _concepts(text: str) -> set[str]:
    lowered = text.lower()
    return {
        concept
        for concept, aliases in CONCEPT_ALIASES.items()
        if any(
            re.search(rf"\b{re.escape(alias)}\b", lowered)
            if len(alias) <= 3
            else alias in lowered
            for alias in aliases
        )
    }


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    union = left_tokens | right_tokens
    token_score = len(left_tokens & right_tokens) / len(union) if union else 0.0
    concept_left = _concepts(left)
    concept_right = _concepts(right)
    concept_union = concept_left | concept_right
    concept_score = (
        len(concept_left & concept_right) / len(concept_union) if concept_union else 0.0
    )
    sequence_score = SequenceMatcher(None, " ".join(sorted(left_tokens)), " ".join(sorted(right_tokens))).ratio()
    return 0.50 * token_score + 0.30 * sequence_score + 0.20 * concept_score


def _best_group(
    label: str,
    groups: list[dict[str, Any]],
    similarity: Any,
) -> tuple[str | None, float]:
    label_concepts = _concepts(label)
    required_entities = label_concepts & {"china", "un", "peninsula"}
    scored: list[tuple[float, str]] = []
    for group in groups:
        text = str(group["text"])
        if required_entities and not required_entities & _concepts(text):
            continue
        score = similarity(label, text)
        if label_concepts & _concepts(text):
            score += 0.20
        scored.append((min(score, 1.0), str(group["group_id"])))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 0.30:
        return None, scored[0][0] if scored else 0.0
    return scored[0][1], scored[0][0]


def _status(score: float, *, embedding: bool = False) -> str:
    full_threshold, partial_threshold = (0.78, 0.60) if embedding else (0.55, 0.30)
    if score >= full_threshold:
        return "supported"
    if score >= partial_threshold:
        return "partially_supported"
    return "missing"


def compare_topic(
    manifest_path: Path,
    nuggets_path: Path,
    topic_id: str,
    *,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    topic = next((row for row in manifest["topics"] if str(row["topic_id"]) == topic_id), None)
    if topic is None:
        raise ValueError(f"topic {topic_id} is absent from {manifest_path}")
    gold_row = next((row for row in _read_jsonl(nuggets_path) if str(row.get("qid")) == topic_id), None)
    if gold_row is None:
        raise ValueError(f"topic {topic_id} is absent from {nuggets_path}")

    groups = list(topic.get("groups") or [])
    claims = list(topic.get("claim_hints") or [])
    claims_by_group: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        claims_by_group.setdefault(str(claim["group_id"]), []).append(claim)

    gold_by_label: dict[str, list[dict[str, Any]]] = {}
    for nugget in gold_row.get("nuggets", []):
        gold_by_label.setdefault(str(nugget["mapped_sub_narrative"]), []).append(nugget)

    embedding_lookup: dict[str, Any] = {}
    if embedding_model:
        from sentence_transformers import SentenceTransformer

        all_texts = [
            str(group["text"])
            for group in groups
        ] + [
            str(claim["text"])
            for claim in claims
        ] + [
            str(nugget["text"])
            for nugget in gold_row.get("nuggets", [])
        ] + list(gold_by_label)
        unique_texts = list(dict.fromkeys(all_texts))
        model = SentenceTransformer(embedding_model, device="cpu")
        vectors = model.encode(unique_texts, normalize_embeddings=True, show_progress_bar=False)
        embedding_lookup = dict(zip(unique_texts, vectors, strict=True))

    def similarity(left: str, right: str) -> float:
        lexical = _lexical_similarity(left, right)
        if not embedding_lookup:
            return lexical
        embedding = float(embedding_lookup[left] @ embedding_lookup[right])
        concept_left = _concepts(left)
        concept_right = _concepts(right)
        concept_union = concept_left | concept_right
        concept = len(concept_left & concept_right) / len(concept_union) if concept_union else 0.0
        return 0.70 * embedding + 0.20 * concept + 0.10 * lexical

    label_rows: list[dict[str, Any]] = []
    nugget_rows: list[dict[str, Any]] = []
    group_by_id = {str(group["group_id"]): group for group in groups}
    gold_index = 0
    for label, label_nuggets in gold_by_label.items():
        group_id, mapping_score = _best_group(label, groups, _lexical_similarity)
        matched = 0
        strict_matched = 0
        for index, nugget in enumerate(label_nuggets, 1):
            nugget_text = str(nugget["text"])
            gold_index += 1
            best_claim = max(
                claims,
                key=lambda claim: similarity(nugget_text, str(claim["text"])),
                default=None,
            )
            best_score = (
                similarity(nugget_text, str(best_claim["text"])) if best_claim else 0.0
            )
            status = _status(best_score, embedding=bool(embedding_model))
            matched += status != "missing"
            strict_matched += status == "supported"
            nugget_rows.append(
                {
                    "nugget_id": f"{topic_id}-N{gold_index:03d}",
                    "label": label,
                    "importance": nugget["importance"],
                    "text": nugget_text,
                    "mapped_group_id": group_id,
                    "best_claim_id": best_claim["claim_id"] if best_claim else None,
                    "best_claim_group_id": best_claim["group_id"] if best_claim else None,
                    "match_score": round(best_score, 4),
                    "status": status,
                }
            )
        label_rows.append(
            {
                "organizer_subnarrative": label,
                "mapped_group_id": group_id,
                "nonagentic_subnarrative": group_by_id[group_id]["text"] if group_id else None,
                "mapping_score": round(mapping_score, 4),
                "organizer_nuggets": len(label_nuggets),
                "matched_nuggets": matched,
                "strict_matched_nuggets": strict_matched,
                "content_covered": bool(matched),
                "covered": bool(group_id),
            }
        )

    all_gold = [str(row["text"]) for row in gold_row["nuggets"]]
    claim_quality_rows: list[dict[str, Any]] = []
    for claim in claims:
        claim_text = str(claim["text"])
        best_score = max((similarity(claim_text, gold) for gold in all_gold), default=0.0)
        claim_quality_rows.append(
            {
                "claim_id": claim["claim_id"],
                "group_id": claim["group_id"],
                "text": claim_text,
                "best_gold_match_score": round(best_score, 4),
                "status": _status(best_score, embedding=bool(embedding_model)),
            }
        )

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(row["status"] for row in rows)
        total = len(rows)
        return {
            "total": total,
            "supported": counts["supported"],
            "partially_supported": counts["partially_supported"],
            "missing": counts["missing"],
            "strict_coverage": round(counts["supported"] / total, 4) if total else 0.0,
            "partial_credit_coverage": round(
                (counts["supported"] + 0.5 * counts["partially_supported"]) / total, 4
            ) if total else 0.0,
        }

    nugget_summary = summary(nugget_rows)
    vital_summary = summary([row for row in nugget_rows if row["importance"] == "vital"])
    quality_summary = summary(claim_quality_rows)
    topic_coverage = sum(row["covered"] for row in label_rows) / len(label_rows) if label_rows else 0.0
    score = 5.0 * (
        0.25 * topic_coverage
        + 0.50 * float(nugget_summary["partial_credit_coverage"])
        + 0.25 * float(quality_summary["partial_credit_coverage"])
    )
    return {
        "schema_version": "nonagentic_2025_comparison_v1",
        "topic_id": topic_id,
        "narrative": topic["narrative"],
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "organizer_nuggets": str(nuggets_path),
            "nonagentic_groups": len(groups),
            "nonagentic_claim_hints": len(claims),
            "organizer_nuggets_count": len(gold_row["nuggets"]),
            "matching": (
                f"offline {embedding_model} embeddings plus lexical/concept similarity; no hosted judge"
                if embedding_model
                else "offline lexical, concept-alias, and sequence similarity; no hosted judge"
            ),
        },
        "overall_score_0_to_5": round(score, 2),
        "subtopic_coverage": {
            "covered": sum(row["covered"] for row in label_rows),
            "total": len(label_rows),
            "rate": round(topic_coverage, 4),
            "rows": label_rows,
        },
        "content_subtopic_coverage": {
            "covered": sum(row["content_covered"] for row in label_rows),
            "total": len(label_rows),
            "rate": round(
                sum(row["content_covered"] for row in label_rows) / len(label_rows), 4
            ) if label_rows else 0.0,
        },
        "nugget_extraction": {
            "all": nugget_summary,
            "vital": vital_summary,
            "rows": nugget_rows,
        },
        "nugget_quality": {
            "generated_claim_precision": quality_summary,
            "rows": claim_quality_rows,
        },
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _repo_root(config_path: Path) -> Path:
    """Find the checkout root so configs work from configs/ or configs/local/."""
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "code").is_dir() and (candidate / "trec-rag-data").exists():
            return candidate
    return config_path.parent


def _render_report(result: dict[str, Any]) -> str:
    sub = result["subtopic_coverage"]
    content_sub = result["content_subtopic_coverage"]
    nuggets = result["nugget_extraction"]
    quality = result["nugget_quality"]["generated_claim_precision"]
    lines = [
        f"# Non-agentic 2025 comparison: topic {result['topic_id']}",
        "",
        "This is an offline diagnostic comparing the released non-agentic handoff with the organizer's 2025 development nuggets. It is not an official TREC score.",
        "",
        f"**Internal score: {result['overall_score_0_to_5']:.2f} / 5**",
        "",
        "## Results",
        "",
        f"- Subtopic coverage: **{sub['covered']}/{sub['total']} ({sub['rate']:.1%})**",
        f"- Content-level subtopic hit rate (any extracted hint): **{content_sub['covered']}/{content_sub['total']} ({content_sub['rate']:.1%})**",
        f"- Nugget extraction, strict: **{nuggets['all']['supported']}/{nuggets['all']['total']} ({nuggets['all']['strict_coverage']:.1%})**",
        f"- Nugget extraction, partial credit: **{nuggets['all']['partial_credit_coverage']:.1%}**",
        f"- Vital nugget strict coverage: **{nuggets['vital']['strict_coverage']:.1%}**",
        f"- Generated-hint quality/precision, partial credit: **{quality['partial_credit_coverage']:.1%}**",
        "",
        "## Subtopic mapping",
        "",
        "| Organizer subnarrative | Non-agentic lane | Covered | Gold nuggets | Matched |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sub["rows"]:
        lane = row["nonagentic_subnarrative"] or "Not represented"
        lines.append(
            f"| {row['organizer_subnarrative']} | {lane} | {'yes' if row['covered'] else 'no'} | {row['organizer_nuggets']} | {row['matched_nuggets']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The non-agentic handoff has good coverage of the broad historical arc, but it compresses the organizer's finer-grained 2025 structure. Unrepresented organizer lanes are the main recall risk; a claim can be well written and still fail evaluation when its subtopic never received a dedicated lane.",
            "",
            "The 0-5 score is a local diagnostic: 25% subtopic coverage, 50% gold-nugget partial-credit coverage, and 25% generated-hint partial-credit precision. The matching is deliberately conservative and deterministic, so a local-model judge could change individual labels but would not be comparable unless the prompt and model are frozen.",
        ]
    )
    return "\n".join(lines) + "\n"


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one topic is required")

    def pooled(section: str, population: str = "all") -> dict[str, Any]:
        rows = [result[section][population] for result in results]
        total = sum(int(row["total"]) for row in rows)
        supported = sum(int(row["supported"]) for row in rows)
        partial = sum(int(row["partially_supported"]) for row in rows)
        missing = sum(int(row["missing"]) for row in rows)
        return {
            "total": total,
            "supported": supported,
            "partially_supported": partial,
            "missing": missing,
            "strict_coverage": round(supported / total, 4) if total else 0.0,
            "partial_credit_coverage": round((supported + 0.5 * partial) / total, 4)
            if total else 0.0,
        }

    topic_rows = []
    for result in results:
        topic_rows.append(
            {
                "topic_id": result["topic_id"],
                "score": result["overall_score_0_to_5"],
                "dedicated_subtopic_rate": result["subtopic_coverage"]["rate"],
                "content_subtopic_rate": result["content_subtopic_coverage"]["rate"],
                "nugget_strict_coverage": result["nugget_extraction"]["all"]["strict_coverage"],
                "nugget_partial_credit_coverage": result["nugget_extraction"]["all"]["partial_credit_coverage"],
                "vital_strict_coverage": result["nugget_extraction"]["vital"]["strict_coverage"],
                "hint_quality_partial_credit": result["nugget_quality"]["generated_claim_precision"]["partial_credit_coverage"],
                "groups": result["inputs"]["nonagentic_groups"],
                "claim_hints": result["inputs"]["nonagentic_claim_hints"],
                "gold_nuggets": result["inputs"]["organizer_nuggets_count"],
            }
        )
    topic_rows.sort(key=lambda row: str(row["topic_id"]))
    mean = lambda key: round(sum(float(row[key]) for row in topic_rows) / len(topic_rows), 4)
    return {
        "schema_version": "nonagentic_2025_comparison_all_topics_v1",
        "topic_count": len(results),
        "topics": topic_rows,
        "macro": {
            "score": round(sum(float(row["score"]) for row in topic_rows) / len(topic_rows), 2),
            "dedicated_subtopic_rate": mean("dedicated_subtopic_rate"),
            "content_subtopic_rate": mean("content_subtopic_rate"),
            "nugget_strict_coverage": mean("nugget_strict_coverage"),
            "nugget_partial_credit_coverage": mean("nugget_partial_credit_coverage"),
            "vital_strict_coverage": mean("vital_strict_coverage"),
            "hint_quality_partial_credit": mean("hint_quality_partial_credit"),
        },
        "pooled": {
            "nuggets": pooled("nugget_extraction"),
            "vital_nuggets": pooled("nugget_extraction", "vital"),
        },
    }


def _render_all_topics_report(aggregate: dict[str, Any]) -> str:
    macro = aggregate["macro"]
    pooled = aggregate["pooled"]
    lines = [
        "# Non-agentic 2025 all-topic comparison",
        "",
        "Offline comparison of the released non-agentic handoff against the organizer's 2025 development nuggets. This is a local diagnostic, not an official TREC score.",
        "",
        f"- Topics: **{aggregate['topic_count']}**",
        f"- Macro internal score: **{macro['score']:.2f}/5**",
        f"- Macro dedicated subtopic coverage: **{macro['dedicated_subtopic_rate']:.1%}**",
        f"- Macro content-level subtopic hit rate: **{macro['content_subtopic_rate']:.1%}**",
        f"- Pooled strict nugget coverage: **{pooled['nuggets']['strict_coverage']:.1%}** ({pooled['nuggets']['supported']}/{pooled['nuggets']['total']})",
        f"- Pooled partial-credit nugget coverage: **{pooled['nuggets']['partial_credit_coverage']:.1%}**",
        f"- Pooled vital strict coverage: **{pooled['vital_nuggets']['strict_coverage']:.1%}**",
        f"- Macro generated-hint quality: **{macro['hint_quality_partial_credit']:.1%}** partial credit",
        "",
        "## Per topic",
        "",
        "| Topic | Score / 5 | Dedicated subtopics | Content hit rate | Nugget strict | Nugget partial | Vital strict | Hint quality |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["topics"]:
        lines.append(
            f"| {row['topic_id']} | {row['score']:.2f} | {row['dedicated_subtopic_rate']:.1%} | {row['content_subtopic_rate']:.1%} | {row['nugget_strict_coverage']:.1%} | {row['nugget_partial_credit_coverage']:.1%} | {row['vital_strict_coverage']:.1%} | {row['hint_quality_partial_credit']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    root = _repo_root(config_path)
    manifest = _resolve(root, str(config["inputs"]["manifest"]))
    nuggets = _resolve(root, str(config["inputs"]["organizer_nuggets"]))
    output = _resolve(root, str(config["output_dir"]))
    embedding_model = str(config["embedding_model"]) if config.get("embedding_model") else None
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    configured_topics = config.get("topic_ids", config.get("topic_id"))
    if configured_topics == "all" or configured_topics is None:
        topic_ids = [str(row["topic_id"]) for row in manifest_payload["topics"]]
    elif isinstance(configured_topics, list):
        topic_ids = [str(topic_id) for topic_id in configured_topics]
    else:
        topic_ids = [str(configured_topics)]
    results = [
        compare_topic(manifest, nuggets, topic_id, embedding_model=embedding_model)
        for topic_id in topic_ids
    ]
    output.mkdir(parents=True, exist_ok=True)
    for result in results:
        (output / f"comparison_{result['topic_id']}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    aggregate = _aggregate_results(results)
    (output / "all_topics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "all_topics_report.md").write_text(
        _render_all_topics_report(aggregate), encoding="utf-8"
    )
    print(output)
    print(f"topics={aggregate['topic_count']}")
    print(f"macro_score={aggregate['macro']['score']:.2f}/5")
    print(f"pooled_nuggets_partial={aggregate['pooled']['nuggets']['partial_credit_coverage']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
