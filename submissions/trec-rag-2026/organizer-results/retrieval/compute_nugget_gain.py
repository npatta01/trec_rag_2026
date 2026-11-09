"""Score a TREC ranked list using document-level nugget qrels."""

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean
from typing import Any


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

WEIGHTS = {"vital": 1.0, "okay": 0.5}
LABELS = {"support", "partial_support", "not_support"}


def score_list(
    docids: list[str], supports: dict[str, set[str]], weights: dict[str, float],
) -> dict:
    """Count each supported nugget once, and every submitted document as effort."""
    if len(docids) != len(set(docids)):
        raise ValueError("Repeated document IDs in a submitted list")
    covered = {nid for docid in docids for nid in supports.get(docid, ())}
    return {
        "effort": len(docids),
        "observed_gain": sum(weights[nid] for nid in sorted(covered)),
        "possible_gain": sum(weights.values()),
    }


def read_run(path: Path, topics: set[str], *, require_all: bool = True) -> dict[str, list[str]]:
    ranked = defaultdict(list)
    run_ids = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields or fields[0] not in topics:
                continue
            if len(fields) != 6:
                raise ValueError(f"{path}:{line_number}: expected six TREC fields")
            run_ids.add(fields[5])
            ranked[fields[0]].append((int(fields[3]), fields[2]))
    if len(run_ids) > 1:
        raise ValueError(f"{path}: expected one run, found multiple run IDs")
    if require_all and set(ranked) != topics:
        raise ValueError(f"{path}: missing topics {sorted(topics - set(ranked))}")
    result = {}
    for qid, rows in ranked.items():
        ranks = [r for r, _ in rows]
        if min(ranks) < 1 or len(ranks) != len(set(ranks)):
            raise ValueError(f"{path}: invalid or duplicate ranks for {qid}")
        result[qid] = [docid for _, docid in sorted(rows)]
    return result


def load_qrels(path: Path) -> tuple[dict[str, list[dict]], dict[str, dict[str, set[str]]]]:
    nuggets_by_topic = {}
    supports = {}
    for row in read_jsonl(path):
        qid, docid, nuggets = (row.get(k) for k in ("qid", "docid", "nuggets"))
        if not all(isinstance(v, str) and v.strip() for v in (qid, docid)):
            raise ValueError(f"{path}: expected nonempty qid and docid strings")
        if not isinstance(nuggets, list) or not nuggets:
            raise ValueError(f"{path}: {qid}/{docid}: expected a nonempty nugget list")
        template, covered, ids = [], set(), set()
        for nugget in nuggets:
            if not isinstance(nugget, dict) or not all(
                isinstance(nugget.get(k), str) and nugget[k].strip()
                for k in ("id", "text", "importance", "assignment")
            ):
                raise ValueError(f"{path}: {qid}/{docid}: invalid nugget fields")
            nid = nugget["id"]
            if nid in ids:
                raise ValueError(f"{path}: {qid}/{docid}: duplicate nugget ID {nid}")
            if nugget["importance"] not in WEIGHTS or nugget["assignment"] not in LABELS:
                raise ValueError(f"{path}: {qid}/{docid}: invalid importance or assignment label")
            ids.add(nid)
            template.append({k: nugget[k] for k in ("id", "text", "importance")})
            if nugget["assignment"] == "support":
                covered.add(nid)
        if qid in nuggets_by_topic and nuggets_by_topic[qid] != template:
            raise ValueError(f"{path}: {qid}/{docid}: inconsistent nugget IDs, text, importance, or order")
        nuggets_by_topic[qid] = template
        documents = supports.setdefault(qid, {})
        if docid in documents:
            raise ValueError(f"{path}: duplicate topic/document pair {qid}/{docid}")
        documents[docid] = covered
    if not nuggets_by_topic:
        raise ValueError(f"{path}: no nugget qrels found")
    return nuggets_by_topic, supports


def topic_sort_key(topic: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.*?)(\d+)", topic)
    return (match[1], int(match[2]), topic) if match else (topic, -1, topic)


def evaluate(run: Path, qrels: Path, *, query: bool = False, complete: bool = False) -> dict:
    return evaluate_loaded(run, *load_qrels(qrels), query=query, complete=complete)


def evaluate_loaded(run: Path, nuggets: dict, supports: dict, *, query: bool = False, complete: bool = False) -> dict:
    """Score a run using qrels already loaded by load_qrels(), for batch reuse."""
    ranked = read_run(run, set(nuggets), require_all=False)
    if not ranked and not complete:
        raise ValueError("No run topics overlap the nugget qrels")
    per_topic = []
    for qid in sorted(nuggets if complete else ranked, key=topic_sort_key):
        weights = {n["id"]: WEIGHTS[n["importance"]] for n in nuggets[qid]}
        scores = score_list(ranked.get(qid, []), supports[qid], weights)
        per_topic.append({
            "qid": qid,
            "gain": scores["observed_gain"] / scores["possible_gain"],
            "effort": scores["effort"],
        })
    result = {
        "overall": {
            "topics": len(per_topic),
            "gain": mean(r["gain"] for r in per_topic),
            "effort": mean(r["effort"] for r in per_topic),
        },
    }
    if query:
        result["per_topic"] = per_topic
    return result


def format_measure(metric: str, value: float | int) -> str:
    count = metric == "effort"
    number = Decimal(str(value)).quantize(Decimal("0.1" if count else "0.0001"), rounding=ROUND_HALF_UP)
    if count:
        return format(number, ".0f" if number == number.to_integral_value() else ".1f")
    return format(number, ".4f")


def render_result(result: dict, metrics: list[str], *, summary: bool, output_format: str) -> str:
    if output_format == "json":
        selected = {}
        if summary:
            selected["overall"] = {"topics": result["overall"]["topics"],
                                   **{metric: result["overall"][metric] for metric in metrics}}
        if "per_topic" in result:
            selected["per_topic"] = [{"qid": row["qid"], **{metric: row[metric] for metric in metrics}}
                                     for row in result["per_topic"]]
        return json.dumps(selected, indent=2, allow_nan=False) + "\n"
    rows = [(row["qid"], row) for row in result.get("per_topic", [])]
    if summary:
        rows.append(("all", result["overall"]))
    return "".join(f"{metric}\t{topic}\t{format_measure(metric, row[metric])}\n"
                   for topic, row in rows for metric in metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "gain: normalized weight of unique supported nuggets (vital=1, okay=0.5). "
        "effort: full submitted document count, including unassessed documents. "
        "Overall scores are topic means. Default output: metric<TAB>topic<TAB>value."))
    parser.add_argument("qrels", type=Path, help="One nugget-assignment JSONL qrels file")
    parser.add_argument("run", type=Path, help="One six-column TREC run file")
    parser.add_argument("-q", action="store_true", help="Include per-topic scores in addition to the summary")
    parser.add_argument("-m", dest="metrics", action="append", choices=("gain", "effort"),
                        help="Select a metric; repeat to select both (default: both)")
    parser.add_argument("-n", action="store_true", help="Suppress the overall summary")
    parser.add_argument("-c", action="store_true",
                        help="Average over all qrels topics; missing run topics contribute zero")
    parser.add_argument("--format", choices=("trec", "json"), default="trec", help="Output format (default: trec)")
    parser.add_argument("--output", type=Path, help="Write the selected output format here instead of stdout")
    args = parser.parse_args()
    try:
        result = evaluate(args.run, args.qrels, query=args.q, complete=args.c)
        output = render_result(result, list(dict.fromkeys(args.metrics or ("gain", "effort"))),
                               summary=not args.n, output_format=args.format)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
