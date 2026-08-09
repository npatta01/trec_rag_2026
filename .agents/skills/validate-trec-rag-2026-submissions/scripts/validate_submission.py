#!/usr/bin/env python3
"""Validate TREC RAG 2026 submission artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Literal, Sequence


CLIMBMIX_DOCUMENT_ID = re.compile(r"shard_\d+_\d+\Z")
MINIMUM_AUTOJUDGE_VERSION = (0, 4, 3)
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class Topic:
    topic_id: str
    narrative: str


@dataclass(frozen=True)
class Finding:
    message: str
    line_number: int | None = None
    topic_id: str | None = None


@dataclass(frozen=True)
class ArtifactResult:
    task: Literal["retrieval", "rag"]
    path: Path
    status: Literal["pass", "pass-with-warnings", "fail"]
    row_count: int | None
    topic_count: int | None
    depth_min: int | None
    depth_max: int | None
    findings: tuple[Finding, ...]
    detail: str = ""


def load_topics(path: Path) -> tuple[Topic, ...]:
    """Load organizer topics from TSV or AutoJudge Request JSONL."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"topics file is not valid UTF-8: {path}") from error

    topics: list[Topic] = []
    seen: set[str] = set()
    is_jsonl = path.suffix.lower() == ".jsonl"

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if is_jsonl:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid topics JSON on line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"topics JSON line {line_number} must be an object"
                )
            topic_id = record.get("request_id")
            narrative = record.get("title")
            if not isinstance(topic_id, str) or not isinstance(narrative, str):
                raise ValueError(
                    f"topics JSON line {line_number} requires string request_id and title"
                )
        else:
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(
                    f"topics TSV line {line_number} must contain exactly two columns"
                )
            topic_id, narrative = fields

        if not topic_id.strip() or not narrative.strip():
            raise ValueError(
                f"topics line {line_number} has an empty topic ID or narrative"
            )
        if topic_id in seen:
            raise ValueError(f"duplicate topic ID: {topic_id}")
        seen.add(topic_id)
        topics.append(Topic(topic_id, narrative))

    if not topics:
        raise ValueError(f"topics file contains no topics: {path}")
    return tuple(topics)


def validate_retrieval(path: Path, topics: tuple[Topic, ...]) -> ArtifactResult:
    """Validate one organizer-facing TREC Retrieval run."""
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        finding = Finding("retrieval run is not valid UTF-8")
        return ArtifactResult(
            "retrieval", path, "fail", None, None, None, None, (finding,)
        )
    except OSError as error:
        finding = Finding(f"could not read retrieval run: {error}")
        return ArtifactResult(
            "retrieval", path, "fail", None, None, None, None, (finding,)
        )

    expected_topics = {topic.topic_id for topic in topics}
    actual_topics: set[str] = set()
    row_count = 0
    depths: dict[str, int] = {}
    documents: dict[str, set[str]] = {}
    previous_scores: dict[str, float] = {}
    run_id: str | None = None
    findings: list[Finding] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row_count += 1
        fields = line.split()
        if len(fields) != 6:
            findings.append(
                Finding("expected six columns", line_number=line_number)
            )
            continue

        topic_id, q0, document_id, rank_text, score_text, current_run_id = fields
        actual_topics.add(topic_id)
        depth = depths.get(topic_id, 0) + 1
        depths[topic_id] = depth

        if q0 != "Q0":
            findings.append(
                Finding(
                    "column 2 must be Q0",
                    line_number=line_number,
                    topic_id=topic_id,
                )
            )

        try:
            rank = int(rank_text)
        except ValueError:
            rank = None
        if rank is None or rank <= 0 or rank != depth:
            findings.append(
                Finding(
                    "ranks must start at 1 and be dense in file order",
                    line_number=line_number,
                    topic_id=topic_id,
                )
            )

        try:
            score = float(score_text)
        except ValueError:
            score = None
            findings.append(
                Finding(
                    "score must be numeric",
                    line_number=line_number,
                    topic_id=topic_id,
                )
            )
        if score is not None:
            if not math.isfinite(score):
                findings.append(
                    Finding(
                        "score must be finite",
                        line_number=line_number,
                        topic_id=topic_id,
                    )
                )
            else:
                previous_score = previous_scores.get(topic_id)
                if previous_score is not None and score > previous_score:
                    findings.append(
                        Finding(
                            "scores must be non-increasing in file order",
                            line_number=line_number,
                            topic_id=topic_id,
                        )
                    )
                previous_scores[topic_id] = score

        if CLIMBMIX_DOCUMENT_ID.fullmatch(document_id) is None:
            findings.append(
                Finding(
                    "invalid ClimbMix document ID",
                    line_number=line_number,
                    topic_id=topic_id,
                )
            )

        topic_documents = documents.setdefault(topic_id, set())
        if document_id in topic_documents:
            findings.append(
                Finding(
                    "duplicate document ID within topic",
                    line_number=line_number,
                    topic_id=topic_id,
                )
            )
        topic_documents.add(document_id)

        if run_id is None:
            run_id = current_run_id
        elif current_run_id != run_id:
            findings.append(
                Finding(
                    f"conflicting run IDs: {run_id!r} and {current_run_id!r}",
                    line_number=line_number,
                    topic_id=topic_id,
                )
            )

    if row_count == 0:
        findings.append(Finding("run contains no rows"))

    missing_topics = sorted(expected_topics - actual_topics)
    if missing_topics:
        findings.append(
            Finding("missing expected topics: " + ", ".join(missing_topics))
        )
    extra_topics = sorted(actual_topics - expected_topics)
    if extra_topics:
        findings.append(
            Finding(
                "topics outside expected population: " + ", ".join(extra_topics)
            )
        )

    topic_depths = tuple(depths.values())
    return ArtifactResult(
        task="retrieval",
        path=path,
        status="fail" if findings else "pass",
        row_count=row_count,
        topic_count=len(actual_topics),
        depth_min=min(topic_depths) if topic_depths else None,
        depth_max=max(topic_depths) if topic_depths else None,
        findings=tuple(findings),
    )


def prepare_autojudge_topics(topics_path: Path, destination: Path) -> Path:
    """Validate topics and convert TSV topics to AutoJudge Request JSONL."""
    topics = load_topics(topics_path)
    if topics_path.suffix.lower() == ".jsonl":
        return topics_path

    destination.write_text(
        "".join(
            json.dumps(
                {"request_id": topic.topic_id, "title": topic.narrative},
                ensure_ascii=False,
            )
            + "\n"
            for topic in topics
        ),
        encoding="utf-8",
    )
    return destination


def _version_at_least_minimum(value: str) -> bool:
    """Compare normalized PEP 440 distribution versions without dependencies."""
    match = re.fullmatch(
        r"v?(?:(\d+)!)?(\d+(?:\.\d+)*)(.*)", value.strip(), re.IGNORECASE
    )
    if match is None:
        return False

    epoch = int(match.group(1) or 0)
    release = tuple(int(part) for part in match.group(2).split("."))
    width = max(len(release), len(MINIMUM_AUTOJUDGE_VERSION))
    release_key = release + (0,) * (width - len(release))
    minimum_key = MINIMUM_AUTOJUDGE_VERSION + (0,) * (
        width - len(MINIMUM_AUTOJUDGE_VERSION)
    )
    if (epoch, release_key) != (0, minimum_key):
        return (epoch, release_key) > (0, minimum_key)

    suffix = match.group(3).lower()
    if not suffix or suffix.startswith("+"):
        return True
    return re.match(
        r"^(?:[-_.]?(?:post|rev|r)\d*(?=$|\+|[-_.]?dev)|-\d+)", suffix
    ) is not None


def build_autojudge_command(strict: bool) -> tuple[str, ...]:
    """Select a compatible local AutoJudge or an isolated uv fallback."""
    try:
        installed_is_compatible = _version_at_least_minimum(
            metadata.version("autojudge-base")
        )
    except metadata.PackageNotFoundError:
        installed_is_compatible = False

    if installed_is_compatible:
        command = (
            sys.executable,
            "-m",
            "autojudge_base.report_tool",
            "check",
        )
    elif shutil.which("uv"):
        command = (
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            "autojudge-base>=0.4.3",
            "python",
            "-m",
            "autojudge_base.report_tool",
            "check",
        )
    else:
        raise RuntimeError(
            "RAG validation requires autojudge-base>=0.4.3 or uv; "
            "install one and rerun this command"
        )

    return command + (("--strict",) if strict else ())


def validate_rag(
    path: Path,
    topics_path: Path,
    *,
    strict: bool,
    runner: Runner = subprocess.run,
) -> ArtifactResult:
    """Validate one RAG report by delegating to the organizer AutoJudge."""
    try:
        command = build_autojudge_command(strict)
    except RuntimeError as error:
        return ArtifactResult(
            "rag",
            path,
            "fail",
            None,
            None,
            None,
            None,
            (Finding(str(error)),),
        )

    try:
        with tempfile.TemporaryDirectory(prefix="rag26-autojudge-") as temporary:
            normalized_topics = prepare_autojudge_topics(
                topics_path, Path(temporary) / "topics.jsonl"
            )
            completed = runner(
                command
                + (
                    str(path),
                    "--spec",
                    "rag26",
                    "--topics",
                    str(normalized_topics),
                ),
                capture_output=True,
                text=True,
                check=False,
            )
    except (OSError, ValueError) as error:
        return ArtifactResult(
            "rag",
            path,
            "fail",
            None,
            None,
            None,
            None,
            (Finding(f"could not run AutoJudge: {error}"),),
        )

    detail = "\n".join(
        section.rstrip()
        for section in (completed.stdout or "", completed.stderr or "")
        if section.strip()
    )
    if completed.returncode != 0:
        status: Literal["pass", "pass-with-warnings", "fail"] = "fail"
        if "JSONDecodeError" in detail:
            location_match = re.search(
                r"JSONDecodeError:[^\n]*line (\d+) column (\d+)", detail
            )
            location = (
                f" at line {location_match.group(1)}, "
                f"column {location_match.group(2)}"
                if location_match
                else ""
            )
            message = (
                f"RAG report is not valid JSONL (AutoJudge JSONDecodeError{location}; "
                f"exit status {completed.returncode})"
            )
        else:
            message = f"AutoJudge exited with status {completed.returncode}"
        findings = (
            Finding(message),
        )
    elif re.search(r"(?m)^\s*SMELL\b", detail):
        status = "pass-with-warnings"
        findings = (Finding("AutoJudge reported SMELL warnings"),)
    else:
        status = "pass"
        findings = ()

    return ArtifactResult(
        task="rag",
        path=path,
        status=status,
        row_count=None,
        topic_count=None,
        depth_min=None,
        depth_max=None,
        findings=findings,
        detail=detail,
    )


def _print_result(result: ArtifactResult) -> None:
    label = result.status.upper().replace("-", " ")
    summary_parts: list[str] = []
    if result.row_count is not None:
        summary_parts.append(f"{result.row_count:,} rows")
    if result.topic_count is not None:
        summary_parts.append(f"{result.topic_count:,} topics")
    if result.depth_min is not None and result.depth_max is not None:
        summary_parts.append(f"depth {result.depth_min:,}-{result.depth_max:,}")
    summary = f" — {'; '.join(summary_parts)}" if summary_parts else ""
    print(f"[{result.task}] {label}: {result.path}{summary}")
    for finding in result.findings:
        location = f"line {finding.line_number}: " if finding.line_number else ""
        print(f"  - {location}{finding.message}")
    if result.detail:
        for line in result.detail.splitlines():
            print(f"  {line}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate TREC RAG 2026 Retrieval and RAG submissions."
    )
    parser.add_argument("--topics", required=True, type=Path)
    parser.add_argument("--retrieval", action="append", default=[], type=Path)
    parser.add_argument("--rag", action="append", default=[], type=Path)
    parser.add_argument(
        "--strict-rag",
        action="store_true",
        help="forward AutoJudge's strict validation flag",
    )
    arguments = parser.parse_args(argv)
    if not arguments.retrieval and not arguments.rag:
        parser.error("provide at least one --retrieval or --rag artifact")

    try:
        topics = load_topics(arguments.topics)
    except (OSError, ValueError) as error:
        print(f"[topics] FAIL: {arguments.topics}")
        print(f"  - {error}")
        return 1

    results = [
        validate_retrieval(path, topics) for path in arguments.retrieval
    ]
    results.extend(
        validate_rag(
            path,
            arguments.topics,
            strict=arguments.strict_rag,
        )
        for path in arguments.rag
    )

    for result in results:
        _print_result(result)
    return 1 if any(result.status == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
