"""Gold-free protected-draft gate prototype.

This deliberately small tool reads only ``draft/final/submission.jsonl`` and
``audit.cards.json`` from a bounded-revision trial root.  It never opens the
handoff, passages, state, manifest, receipts, or any post-hoc evaluation data.

The gate is a contract check, not a quality judge: an invalid replacement index
rejects an arm, while deterministic sentence anchors report how much untargeted
draft content remains traceable after revision.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


MIN_INFO_TOKEN_LENGTH = 6
MIN_SHARED_INFO_TOKENS = 2
MIN_INFO_COVERAGE = 0.35
MIN_SHARED_NUMBER_TOKENS = 1
MIN_SHARED_CITATIONS = 1
MIN_ANCHOR_RATE = 0.90

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\d+(?:[,.]\d+)?(?:\s*[-–]\s*\d+(?:[,.]\d+)?)?%?)"
    r"(?![A-Za-z0-9])"
)
_STOPWORDS = frozenset(
    "a an and are as at be because been being but by can could did do does for from "
    "had has have how i if in into is it its may more most of on one or our should "
    "so than that the their them there these they this to two under was we were what "
    "when where which while who will with would you your"
    .split()
)


@dataclass(frozen=True)
class Arm:
    topic_id: str
    references: tuple[str, ...]
    answer: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class InvalidReplacementIndex:
    group_id: str
    card_index: int
    value: object
    reason: str


@dataclass(frozen=True)
class AnchorMatch:
    draft_index: int
    final_index: int
    shared_info_tokens: int
    info_coverage: float
    shared_numbers: int
    shared_citations: int


@dataclass(frozen=True)
class GateReport:
    topic_id: str
    draft_answer_objects: int
    final_answer_objects: int
    audit_cards: int
    valid_replacement_indices: tuple[int, ...]
    invalid_replacement_indices: tuple[InvalidReplacementIndex, ...]
    untargeted_answer_objects: int
    strict_exact_preserved: int
    strict_exact_rate: float
    anchor_preserved: int
    anchor_rate: float
    anchor_threshold: float
    anchor_margin: float
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "draft_answer_objects": self.draft_answer_objects,
            "final_answer_objects": self.final_answer_objects,
            "audit_cards": self.audit_cards,
            "valid_replacement_indices": list(self.valid_replacement_indices),
            "invalid_replacement_indices": [
                {
                    "group_id": item.group_id,
                    "card_index": item.card_index,
                    "value": item.value,
                    "reason": item.reason,
                }
                for item in self.invalid_replacement_indices
            ],
            "untargeted_answer_objects": self.untargeted_answer_objects,
            "strict_exact_preserved": self.strict_exact_preserved,
            "strict_exact_rate": self.strict_exact_rate,
            "anchor_preserved": self.anchor_preserved,
            "anchor_rate": self.anchor_rate,
            "anchor_threshold": self.anchor_threshold,
            "anchor_margin": self.anchor_margin,
            "decision": self.decision,
        }


def _read_one_jsonl(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"{path}: expected exactly one JSON object")
    return rows[0]


def load_arm(path: Path) -> Arm:
    record = _read_one_jsonl(path)
    metadata = record.get("metadata")
    answer = record.get("answer")
    references = record.get("references")
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("narrative_id"), str)
        or not isinstance(references, list)
        or not all(isinstance(item, str) for item in references)
        or not isinstance(answer, list)
        or not all(isinstance(item, dict) and isinstance(item.get("text"), str) for item in answer)
    ):
        raise ValueError(f"{path}: unsupported submission shape")
    return Arm(
        topic_id=metadata["narrative_id"],
        references=tuple(references),
        answer=tuple(answer),
    )


def load_audit_cards(path: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: audit cards must be an object")
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for group_id, cards in value.items():
        if not isinstance(group_id, str) or not isinstance(cards, list):
            raise ValueError(f"{path}: malformed audit group {group_id!r}")
        if not all(isinstance(card, dict) for card in cards):
            raise ValueError(f"{path}: malformed card in group {group_id!r}")
        output[group_id] = tuple(cards)
    return output


def replacement_indices(
    cards_by_group: dict[str, tuple[dict[str, Any], ...]],
    *,
    draft_answer_count: int,
) -> tuple[tuple[int, ...], tuple[InvalidReplacementIndex, ...], int]:
    """Return valid target indices, surfaced invalid indices, and card count."""

    valid: set[int] = set()
    invalid: list[InvalidReplacementIndex] = []
    card_count = 0
    for group_id, cards in cards_by_group.items():
        for card_index, card in enumerate(cards, start=1):
            card_count += 1
            value = card.get("replacement_answer_index")
            if value is None:
                continue
            if type(value) is not int:
                invalid.append(
                    InvalidReplacementIndex(group_id, card_index, value, "not a zero-based integer")
                )
            elif value < 0 or value >= draft_answer_count:
                invalid.append(
                    InvalidReplacementIndex(
                        group_id,
                        card_index,
                        value,
                        f"outside draft answer range 0..{max(0, draft_answer_count - 1)}",
                    )
                )
            else:
                valid.add(value)
    return tuple(sorted(valid)), tuple(invalid), card_count


def _high_information_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token.casefold().strip("'-")
        for token in _TOKEN_RE.findall(text)
        if len(token.strip("'-")) >= MIN_INFO_TOKEN_LENGTH
        and token.casefold() not in _STOPWORDS
    )


def _numbers(text: str) -> frozenset[str]:
    return frozenset(match.replace(",", "").replace(" ", "") for match in _NUMBER_RE.findall(text))


def _citation_docids(arm: Arm, answer: dict[str, Any]) -> frozenset[str]:
    citations = answer.get("citations", ())
    if not isinstance(citations, list):
        return frozenset()
    return frozenset(
        arm.references[index]
        for index in citations
        if type(index) is int and 0 <= index < len(arm.references)
    )


def _anchor_score(
    draft: Arm,
    draft_answer: dict[str, Any],
    final: Arm,
    final_answer: dict[str, Any],
) -> tuple[int, float, int, int]:
    draft_tokens = _high_information_tokens(draft_answer["text"])
    final_tokens = _high_information_tokens(final_answer["text"])
    shared_info = len(draft_tokens & final_tokens)
    coverage = shared_info / max(1, min(len(draft_tokens), len(final_tokens)))
    shared_numbers = len(_numbers(draft_answer["text"]) & _numbers(final_answer["text"]))
    shared_citations = len(
        _citation_docids(draft, draft_answer) & _citation_docids(final, final_answer)
    )
    return shared_info, coverage, shared_numbers, shared_citations


def _anchor_eligible(score: tuple[int, float, int, int]) -> bool:
    shared_info, coverage, shared_numbers, shared_citations = score
    return (
        shared_info >= MIN_SHARED_INFO_TOKENS
        and coverage >= MIN_INFO_COVERAGE
    ) or (
        shared_numbers >= MIN_SHARED_NUMBER_TOKENS
        and shared_info >= 1
        and shared_citations >= MIN_SHARED_CITATIONS
    ) or (
        shared_citations >= MIN_SHARED_CITATIONS
        and shared_info >= 3
        and coverage >= 0.20
    )


def anchor_matches(
    draft: Arm,
    final: Arm,
    *,
    excluded_draft_indices: Iterable[int],
) -> tuple[AnchorMatch, ...]:
    excluded = set(excluded_draft_indices)
    candidates: list[tuple[tuple[int, float, int, int], int, int]] = []
    for draft_index, draft_answer in enumerate(draft.answer):
        if draft_index in excluded:
            continue
        for final_index, final_answer in enumerate(final.answer):
            score = _anchor_score(draft, draft_answer, final, final_answer)
            if _anchor_eligible(score):
                candidates.append((score, draft_index, final_index))
    candidates.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], -item[0][3], item[1], item[2]))
    used_draft: set[int] = set()
    used_final: set[int] = set()
    matches: list[AnchorMatch] = []
    for score, draft_index, final_index in candidates:
        if draft_index in used_draft or final_index in used_final:
            continue
        used_draft.add(draft_index)
        used_final.add(final_index)
        matches.append(
            AnchorMatch(
                draft_index=draft_index,
                final_index=final_index,
                shared_info_tokens=score[0],
                info_coverage=score[1],
                shared_numbers=score[2],
                shared_citations=score[3],
            )
        )
    return tuple(sorted(matches, key=lambda item: item.draft_index))


def strict_exact_preservation(
    draft: Arm,
    final: Arm,
    *,
    excluded_draft_indices: Iterable[int],
) -> tuple[int, int]:
    excluded = set(excluded_draft_indices)
    remaining_final = Counter(answer["text"] for answer in final.answer)
    preserved = 0
    total = 0
    for index, answer in enumerate(draft.answer):
        if index in excluded:
            continue
        total += 1
        text = answer["text"]
        if remaining_final[text]:
            preserved += 1
            remaining_final[text] -= 1
    return preserved, total


def evaluate_root(root: Path) -> GateReport:
    draft = load_arm(root / "evaluation" / "draft" / "submission.jsonl")
    final = load_arm(root / "evaluation" / "final" / "submission.jsonl")
    if draft.topic_id != final.topic_id:
        raise ValueError(f"{root}: draft/final topic IDs differ")
    cards = load_audit_cards(root / "audit.cards.json")
    valid_indices, invalid_indices, card_count = replacement_indices(
        cards, draft_answer_count=len(draft.answer)
    )
    exact, untargeted = strict_exact_preservation(
        draft, final, excluded_draft_indices=valid_indices
    )
    matches = anchor_matches(draft, final, excluded_draft_indices=valid_indices)
    anchor_rate = len(matches) / max(1, untargeted)
    anchor_margin = anchor_rate - MIN_ANCHOR_RATE
    if invalid_indices:
        decision = "REJECT_INVALID_AUDIT_INDEX"
    elif anchor_rate < MIN_ANCHOR_RATE:
        decision = "REJECT_LOW_ANCHOR_PRESERVATION"
    else:
        decision = "ALLOW_PROTECTED_DRAFT"
    return GateReport(
        topic_id=draft.topic_id,
        draft_answer_objects=len(draft.answer),
        final_answer_objects=len(final.answer),
        audit_cards=card_count,
        valid_replacement_indices=valid_indices,
        invalid_replacement_indices=invalid_indices,
        untargeted_answer_objects=untargeted,
        strict_exact_preserved=exact,
        strict_exact_rate=exact / max(1, untargeted),
        anchor_preserved=len(matches),
        anchor_rate=anchor_rate,
        anchor_threshold=MIN_ANCHOR_RATE,
        anchor_margin=anchor_margin,
        decision=decision,
    )


def _format_report(report: GateReport) -> str:
    invalid = "none"
    if report.invalid_replacement_indices:
        invalid = "; ".join(
            f"{item.group_id}[card {item.card_index}]={item.value!r} ({item.reason})"
            for item in report.invalid_replacement_indices
        )
    return "\n".join(
        [
            f"topic={report.topic_id} decision={report.decision}",
            f"arms=draft:{report.draft_answer_objects},final:{report.final_answer_objects},audit_cards:{report.audit_cards}",
            f"valid_replacement_indices={list(report.valid_replacement_indices)}",
            f"invalid_replacement_indices={invalid}",
            f"strict_untargeted_exact={report.strict_exact_preserved}/{report.untargeted_answer_objects} ({report.strict_exact_rate:.3f})",
            f"paraphrase_anchor={report.anchor_preserved}/{report.untargeted_answer_objects} ({report.anchor_rate:.3f}); threshold:{report.anchor_threshold:.3f}; margin:{report.anchor_margin:+.3f}",
        ]
    )


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        nargs="+",
        type=Path,
        required=True,
        metavar="TRIAL_ROOT",
        help="One or more bounded-revision private trial roots.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    reports = [evaluate_root(root.expanduser().resolve()) for root in args.batch]
    if args.json:
        print(json.dumps([report.to_dict() for report in reports], indent=2, sort_keys=True))
        return
    print(
        "thresholds="
        f"info_token_length>={MIN_INFO_TOKEN_LENGTH},"
        f"shared_info>={MIN_SHARED_INFO_TOKENS},"
        f"info_coverage>={MIN_INFO_COVERAGE:.2f},"
        f"anchor_rate>={MIN_ANCHOR_RATE:.2f}"
    )
    for report in reports:
        print(_format_report(report))
    allowed = [report.topic_id for report in reports if report.decision == "ALLOW_PROTECTED_DRAFT"]
    rejected = [report.topic_id for report in reports if report.decision != "ALLOW_PROTECTED_DRAFT"]
    print(f"batch_allowed={allowed}; batch_rejected={rejected}")
    if any(report.invalid_replacement_indices for report in reports):
        print(
            "interpretation=the batch is contract-separable, but the rejection is driven by "
            "an invalid audit index; strict and paraphrase anchors do not distinguish quality"
        )
    else:
        print("interpretation=strict and paraphrase anchors do not provide a contract rejection signal")
    print(
        "recommendation=use by-construction patch/edit-operation identity and replacement-index "
        "validation for shadow gating; treat content anchors as diagnostics, not a quality proof"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"error: {type(exc).__name__}: {exc}") from exc
