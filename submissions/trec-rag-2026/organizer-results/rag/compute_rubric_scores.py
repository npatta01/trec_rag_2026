"""Score generated answers using rubric/nugget support assignments."""

import argparse
from collections.abc import Iterator
from typing import Any
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import json
from pathlib import Path
import re

METRICS = ("vital_strict", "all_strict")
LABELS = {"support", "partial_support", "not_support"}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def score_items(nuggets):
    """Return exact fractions. Zero denominators score zero by release convention."""
    if not isinstance(nuggets, list):
        raise ValueError("Expected a list of rubric items")
    for n in nuggets:
        if not isinstance(n, dict) or n.get("importance") not in {"vital", "okay"} or n.get("assignment") not in LABELS:
            raise ValueError("Invalid importance or unresolved assignment label")
    vital = [n for n in nuggets if n["importance"] == "vital"]
    def coverage(items):
        return Fraction(sum(n["assignment"] == "support" for n in items), len(items)) if items else Fraction()
    return {"vital_strict": coverage(vital), "all_strict": coverage(nuggets)}


def topic_sort_key(topic):
    match = re.fullmatch(r"(.*?)(\d+)", topic)
    return (match[1], int(match[2]), topic) if match else (topic, -1, topic)


def read_topics(path: Path) -> set[str]:
    """Read one topic ID per line; ignore blank lines, reject duplicates."""
    topics = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        topic = line.strip()
        if not topic:
            continue
        if topic == "all" or len(topic.split()) != 1:
            raise ValueError(f"{path}:{line_number}: expected one topic ID, other than all")
        if topic in topics:
            raise ValueError(f"{path}:{line_number}: duplicate topic {topic}")
        topics.add(topic)
    if not topics:
        raise ValueError(f"{path}: topic list is empty")
    return topics


def load_assignments(path, *, runs=None, topics=None, complete=False):
    """Validate and score one collection.

    Use the supplied topic list, or the topics found in the selected assignments.
    Missing topics are errors unless complete=True and a topic list is supplied;
    then missing topics score zero. Unknown runs remain errors.
    """
    runs = set(runs) if runs is not None else None
    topics = set(topics) if topics is not None else None
    if complete and topics is None:
        raise ValueError("complete=True requires an explicit topic set")
    if topics is not None and not topics:
        raise ValueError("Topic set is empty")
    values, templates = {}, {}
    for row in read_jsonl(Path(path)):
        run, topic = row.get("run_id"), row.get("qid")
        if not all(isinstance(v, str) and v.strip() for v in (run, topic)) or topic == "all":
            raise ValueError(f"{path}: expected nonempty run_id and qid; qid cannot be all")
        if topics is not None and topic not in topics:
            continue
        if runs is not None and run not in runs:
            continue
        key = run, topic
        if key in values:
            raise ValueError(f"{path}: duplicate run/topic {key}")
        nuggets = row.get("nuggets")
        scores = score_items(nuggets)
        if not nuggets or any(not isinstance(n.get("text"), str) or not n["text"].strip() for n in nuggets):
            raise ValueError(f"{path}: {run}/{topic}: expected a nonempty rubric with item text")
        template = [(n.get("id"), n["text"], n["importance"]) for n in nuggets]
        ids = [n["id"] for n in nuggets if "id" in n]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: {run}/{topic}: duplicate rubric item IDs")
        if templates.setdefault(topic, template) != template:
            raise ValueError(f"{path}: {topic}: inconsistent rubric IDs, text, importance, or order")
        values[key] = scores
    if not values:
        raise ValueError("No selected assignments")
    expected_runs = runs if runs is not None else {r for r, _ in values}
    absent_runs = expected_runs - {r for r, _ in values}
    if absent_runs:
        raise ValueError(f"No selected assignments for runs: {sorted(absent_runs)}")
    expected_topics = topics if topics is not None else {t for _, t in values}
    missing = {(r, t) for r in expected_runs for t in expected_topics} - set(values)
    if missing and not complete:
        raise ValueError(f"Missing {len(missing)} run/topic assignments; examples: {sorted(missing)[:5]}")
    for key in missing:
        values[key] = {metric: Fraction() for metric in METRICS}
    for run in expected_runs:
        values[run, "all"] = {metric: sum((values[run, t][metric] for t in expected_topics), Fraction()) / len(expected_topics)
                              for metric in METRICS}
    return values


def evaluate(assignments: Path, *, run: str | None = None, query: bool = False,
             complete: bool = False, topics=None) -> dict:
    values = load_assignments(assignments, runs=[run] if run is not None else None,
                              topics=topics, complete=complete)
    return evaluate_loaded(values, run=run, query=query)


def evaluate_loaded(values: dict, *, run: str | None = None, query: bool = False) -> dict:
    """Score one run using assignments already loaded, for batch reuse."""
    runs = {r for r, _ in values}
    if run is None:
        if len(runs) != 1:
            raise ValueError("Expected one run; use --run ID to select a run")
        run = next(iter(runs))
    if run not in runs:
        raise ValueError(f"No assignments for run {run}")
    topics = sorted((t for r, t in values if r == run and t != "all"), key=topic_sort_key)
    result = {"overall": {"topics": len(topics), **values[run, "all"]}}
    if query:
        result["per_topic"] = [{"qid": t, **values[run, t]} for t in topics]
    return result


def format_measure(metric: str, value) -> str:
    # Keep exact fractions until final formatting, including half-rounding cases.
    number = Decimal(value.numerator) / Decimal(value.denominator) if isinstance(value, Fraction) else Decimal(str(value))
    return format(number.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP), ".4f")


def render_result(result: dict, metrics: list[str], *, summary: bool, output_format: str) -> str:
    if output_format == "json":
        selected = {}
        if summary:
            selected["overall"] = {"topics": result["overall"]["topics"],
                                   **{metric: float(result["overall"][metric]) for metric in metrics}}
        if "per_topic" in result:
            selected["per_topic"] = [{"qid": row["qid"], **{metric: float(row[metric]) for metric in metrics}}
                                     for row in result["per_topic"]]
        return json.dumps(selected, indent=2, allow_nan=False) + "\n"
    rows = [(row["qid"], row) for row in result.get("per_topic", [])]
    if summary:
        rows.append(("all", result["overall"]))
    return "".join(f"{metric}\t{topic}\t{format_measure(metric, row[metric])}\n"
                   for topic, row in rows for metric in metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "vital_strict: fully supported vital items / all vital items. "
        "all_strict: fully supported items / all vital and okay items. "
        "Partial support earns zero. Overall scores are topic means. "
        "Default output: metric<TAB>topic<TAB>value."))
    parser.add_argument("assignments", type=Path, help="One answer-assignment JSONL file")
    parser.add_argument("--run", required=True, help="Run ID to score, matching run_id in the assignment file")
    parser.add_argument("-q", action="store_true", help="Include per-topic scores in addition to the summary")
    parser.add_argument("-c", action="store_true",
                        help="Average over all topics in --topics; missing topics for the selected run score zero")
    parser.add_argument("--topics", type=Path,
                        help="Evaluation topic list (one ID per line); required with -c. "
                             "Without -c, missing listed topics are errors")
    parser.add_argument("-m", dest="metrics", action="append", choices=METRICS,
                        help="Select a metric; repeat to select both (default: both)")
    parser.add_argument("-n", action="store_true", help="Suppress the overall summary")
    parser.add_argument("--format", choices=("trec", "json"), default="trec", help="Output format (default: trec)")
    parser.add_argument("--output", type=Path, help="Write the selected output format here instead of stdout")
    args = parser.parse_args()
    if args.c and args.topics is None:
        parser.error("-c requires --topics PATH")
    try:
        topics = read_topics(args.topics) if args.topics is not None else None
        result = evaluate(args.assignments, run=args.run, query=args.q, complete=args.c, topics=topics)
        output = render_result(result, list(dict.fromkeys(args.metrics or METRICS)),
                               summary=not args.n, output_format=args.format)
        if args.output:
            inputs = [args.assignments, *([args.topics] if args.topics is not None else [])]
            if args.output.resolve() in {path.resolve() for path in inputs}:
                raise ValueError("Output must not overwrite an input file")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
