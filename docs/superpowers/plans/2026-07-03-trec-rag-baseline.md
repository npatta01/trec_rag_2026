# TREC RAG Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable TREC RAG 2026 baseline that converts dev topics, retrieves ClimbMix BM25 results, writes retrieval and RAG outputs, and validates both files.

**Architecture:** Keep the baseline as three focused CLI scripts under `scripts/`: one converter, one runner, and one validator. The runner uses standard-library HTTP calls and deterministic evidence extraction so tests can run without network access.

**Tech Stack:** Python 3 standard library, `pytest` for unit tests, Pyserini REST API over HTTP, JSONL/TSV output files.

---

## File Structure

- Create `scripts/__init__.py`: marks `scripts` as an importable local package for tests.
- Create `scripts/convert_dev_topics.py`: TSV-to-official-topic-JSONL converter.
- Create `scripts/run_baseline.py`: topic loading, Pyserini REST client, cache writing, retrieval TSV writing, and deterministic RAG JSONL writing.
- Create `scripts/validate_outputs.py`: retrieval and RAG output validation CLI.
- Create `tests/test_convert_dev_topics.py`: converter unit tests.
- Create `tests/test_run_baseline.py`: runner helper unit tests with fake candidates.
- Create `tests/test_validate_outputs.py`: validator acceptance and failure tests.
- Create `.gitignore`: excludes local secrets, cache, generated output, and Python cache files.
- Create `README.md`: documents exact commands for converting, running, validating, and smoke testing.

## Task 1: Converter Tests And Implementation

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/convert_dev_topics.py`
- Test: `tests/test_convert_dev_topics.py`

- [ ] **Step 1: Write the failing converter tests**

Create `tests/test_convert_dev_topics.py`:

```python
import json

from scripts.convert_dev_topics import convert_tsv_to_jsonl, derive_title


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_derive_title_uses_first_words_and_normalizes_space():
    text = "  This is a long narrative about industrialization and migration across countries.  "

    assert derive_title(text, max_words=6) == "This is a long narrative about"


def test_convert_tsv_to_jsonl_preserves_id_and_text(tmp_path):
    source = tmp_path / "topics.tsv"
    output = tmp_path / "topics.jsonl"
    source.write_text(
        "14\tWhat were the causes and effects of the Industrial Revolution?\n"
        "22\tExplain inmate rights, rehabilitation, and recidivism.\n",
        encoding="utf-8",
    )

    count = convert_tsv_to_jsonl(source, output, title_words=5)

    assert count == 2
    assert read_jsonl(output) == [
        {
            "id": "14",
            "title": "What were the causes and",
            "narrative": "What were the causes and effects of the Industrial Revolution?",
        },
        {
            "id": "22",
            "title": "Explain inmate rights rehabilitation and",
            "narrative": "Explain inmate rights, rehabilitation, and recidivism.",
        },
    ]


def test_convert_tsv_to_jsonl_rejects_bad_rows(tmp_path):
    source = tmp_path / "bad.tsv"
    output = tmp_path / "topics.jsonl"
    source.write_text("missing-tab-row\n", encoding="utf-8")

    try:
        convert_tsv_to_jsonl(source, output)
    except ValueError as exc:
        assert "line 1" in str(exc)
        assert "qid<TAB>text" in str(exc)
    else:
        raise AssertionError("Expected malformed TSV row to fail")
```

- [ ] **Step 2: Run converter tests and verify they fail**

Run:

```bash
pytest tests/test_convert_dev_topics.py -q
```

Expected: tests fail with `ModuleNotFoundError` or missing `scripts.convert_dev_topics` because the converter does not exist yet.

- [ ] **Step 3: Add the converter implementation**

Create `scripts/__init__.py`:

```python
"""Local command modules for the TREC RAG baseline."""
```

Create `scripts/convert_dev_topics.py`:

```python
#!/usr/bin/env python3
"""Convert local TREC RAG development TSV topics to official-style JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def derive_title(text: str, max_words: int = 12) -> str:
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", " ".join(text.split()))
    if not words:
        return "Untitled topic"
    return " ".join(words[:max_words])


def convert_tsv_to_jsonl(input_path: Path, output_path: Path, title_words: int = 12) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as sink:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"line {line_number}: expected qid<TAB>text")

            qid, text = line.split("\t", 1)
            qid = qid.strip()
            narrative = " ".join(text.split())
            if not qid or not narrative:
                raise ValueError(f"line {line_number}: expected non-empty qid<TAB>text")

            record = {
                "id": qid,
                "title": derive_title(narrative, max_words=title_words),
                "narrative": narrative,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert qid<TAB>text development topics to TREC RAG topic JSONL."
    )
    parser.add_argument("input_tsv", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--title-words", type=int, default=12)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    count = convert_tsv_to_jsonl(args.input_tsv, args.output_jsonl, args.title_words)
    print(f"Wrote {count} topics to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run converter tests and verify they pass**

Run:

```bash
pytest tests/test_convert_dev_topics.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit converter work**

Run:

```bash
git add scripts/__init__.py scripts/convert_dev_topics.py tests/test_convert_dev_topics.py
git commit -m "feat: add dev topic converter"
```

Expected: commit succeeds.

## Task 2: Baseline Runner Helper Tests And Implementation

**Files:**
- Create: `scripts/run_baseline.py`
- Test: `tests/test_run_baseline.py`

- [ ] **Step 1: Write failing runner helper tests**

Create `tests/test_run_baseline.py`:

```python
import json

from scripts.run_baseline import (
    Candidate,
    Topic,
    build_rag_object,
    extract_doc_text,
    load_topics,
    normalize_candidates,
    write_retrieval_run,
)


def test_load_topics_requires_official_fields(tmp_path):
    topics_path = tmp_path / "topics.jsonl"
    topics_path.write_text(
        json.dumps({"id": "1", "title": "Industrial Revolution", "narrative": "Explain causes."})
        + "\n",
        encoding="utf-8",
    )

    assert load_topics(topics_path) == [
        Topic(id="1", title="Industrial Revolution", narrative="Explain causes.")
    ]


def test_extract_doc_text_prefers_common_text_fields():
    doc = {"title": "Ignored heading", "contents": "Main body text from the document."}

    assert extract_doc_text(doc) == "Main body text from the document."


def test_normalize_candidates_uses_returned_rank_and_score():
    response = {
        "candidates": [
            {
                "docid": "shard_1_1",
                "rank": 2,
                "score": 12.5,
                "doc": {"contents": "Evidence sentence one. Evidence sentence two."},
            }
        ]
    }

    assert normalize_candidates(response) == [
        Candidate(
            docid="shard_1_1",
            rank=2,
            score=12.5,
            text="Evidence sentence one. Evidence sentence two.",
        )
    ]


def test_write_retrieval_run_uses_six_column_format(tmp_path):
    output = tmp_path / "r.tsv"
    topic = Topic(id="7", title="Topic", narrative="Narrative")
    candidates = [
        Candidate(docid="shard_1_1", rank=1, score=10.0, text="Alpha."),
        Candidate(docid="shard_1_2", rank=2, score=9.0, text="Beta."),
    ]

    write_retrieval_run({topic.id: candidates}, output, run_id="test_run")

    assert output.read_text(encoding="utf-8").splitlines() == [
        "7 Q0 shard_1_1 1 10 test_run",
        "7 Q0 shard_1_2 2 9 test_run",
    ]


def test_build_rag_object_cites_selected_references():
    topic = Topic(id="7", title="Topic", narrative="Narrative")
    candidates = [
        Candidate(docid="shard_1_1", rank=1, score=10.0, text="Alpha evidence sentence."),
        Candidate(docid="shard_1_2", rank=2, score=9.0, text="Beta evidence sentence."),
    ]

    rag = build_rag_object(topic, candidates, team_id="team", run_id="run", evidence_limit=2)

    assert rag["metadata"]["type"] == "automatic"
    assert rag["metadata"]["narrative_id"] == "7"
    assert rag["references"] == ["shard_1_1", "shard_1_2"]
    assert rag["answer"] == [
        {"text": "Alpha evidence sentence.", "citations": [0]},
        {"text": "Beta evidence sentence.", "citations": [1]},
    ]


def test_build_rag_object_reports_insufficient_evidence_without_references():
    topic = Topic(id="7", title="Topic", narrative="Narrative")

    rag = build_rag_object(topic, [], team_id="team", run_id="run")

    assert rag["references"] == []
    assert rag["answer"] == [
        {
            "text": "The retrieved evidence did not provide enough usable text to answer this topic.",
            "citations": [],
        }
    ]
```

- [ ] **Step 2: Run runner helper tests and verify they fail**

Run:

```bash
pytest tests/test_run_baseline.py -q
```

Expected: tests fail with missing `scripts.run_baseline`.

- [ ] **Step 3: Add runner helpers and output writers**

Create `scripts/run_baseline.py` with these imports, data structures, topic loading, document text extraction, candidate normalization, retrieval writing, and RAG writing helpers:

```python
#!/usr/bin/env python3
"""Run a conservative Pyserini/ClimbMix baseline for TREC RAG 2026."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://api.castorini.uwaterloo.ca"
DEFAULT_INDEX = "climbmix-400b"
DEFAULT_HITS = 100
DEFAULT_RUN_ID = "pyserini_climbmix_bm25_top100"
DEFAULT_TEAM_ID = "baseline-team"
DEFAULT_PROMPT = (
    "Deterministic extractive baseline: use top-ranked retrieved ClimbMix document text "
    "and cite each emitted evidence sentence."
)


@dataclass(frozen=True)
class Topic:
    id: str
    title: str
    narrative: str


@dataclass(frozen=True)
class Candidate:
    docid: str
    rank: int
    score: float
    text: str


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_topics(path: Path) -> list[Topic]:
    topics: list[Topic] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number}: invalid JSON") from exc
            missing = [field for field in ("id", "title", "narrative") if not record.get(field)]
            if missing:
                raise ValueError(f"{path}: line {line_number}: missing fields {missing}")
            topics.append(
                Topic(
                    id=str(record["id"]),
                    title=str(record["title"]),
                    narrative=str(record["narrative"]),
                )
            )
    if not topics:
        raise ValueError(f"{path}: no topics found")
    return topics


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def extract_doc_text(doc: Any) -> str:
    if doc is None:
        return ""
    if isinstance(doc, str):
        return _normalize_space(doc)
    if isinstance(doc, dict):
        for key in ("contents", "text", "body", "passage", "abstract"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_space(value)
        parts = [extract_doc_text(value) for value in doc.values()]
        return _normalize_space(" ".join(part for part in parts if part))
    if isinstance(doc, list):
        parts = [extract_doc_text(value) for value in doc]
        return _normalize_space(" ".join(part for part in parts if part))
    return ""


def normalize_candidates(response: dict[str, Any]) -> list[Candidate]:
    raw_candidates = response.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("search response missing candidates list")

    candidates: list[Candidate] = []
    for fallback_rank, raw_candidate in enumerate(raw_candidates, start=1):
        if not isinstance(raw_candidate, dict):
            continue
        docid = raw_candidate.get("docid")
        if not docid:
            continue
        rank = int(raw_candidate.get("rank") or fallback_rank)
        score = float(raw_candidate.get("score") or 0.0)
        text = extract_doc_text(raw_candidate.get("doc"))
        candidates.append(Candidate(docid=str(docid), rank=rank, score=score, text=text))
    return candidates


def _format_score(score: float) -> str:
    return f"{score:.10g}"


def write_retrieval_run(
    results_by_topic: dict[str, list[Candidate]], output_path: Path, run_id: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        for topic_id, candidates in results_by_topic.items():
            for rank, candidate in enumerate(candidates, start=1):
                sink.write(
                    f"{topic_id} Q0 {candidate.docid} {rank} "
                    f"{_format_score(candidate.score)} {run_id}\n"
                )


def _first_sentence(text: str, max_chars: int = 360) -> str:
    clean = _normalize_space(text)
    if not clean:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", clean)
    sentence = match.group(1) if match else clean
    if len(sentence) <= max_chars:
        return sentence
    truncated = sentence[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{truncated}."


def build_rag_object(
    topic: Topic,
    candidates: list[Candidate],
    team_id: str,
    run_id: str,
    evidence_limit: int = 3,
) -> dict[str, Any]:
    references: list[str] = []
    answer: list[dict[str, Any]] = []

    for candidate in candidates:
        if len(references) >= evidence_limit:
            break
        sentence = _first_sentence(candidate.text)
        if not sentence:
            continue
        references.append(candidate.docid)
        answer.append({"text": sentence, "citations": [len(references) - 1]})

    if not answer:
        answer = [
            {
                "text": "The retrieved evidence did not provide enough usable text to answer this topic.",
                "citations": [],
            }
        ]

    return {
        "metadata": {
            "team_id": team_id,
            "run_id": run_id,
            "type": "automatic",
            "narrative_id": topic.id,
            "title": topic.title,
            "narrative": topic.narrative,
            "prompt": DEFAULT_PROMPT,
        },
        "references": references,
        "answer": answer,
    }


def write_rag_output(
    topics: list[Topic],
    results_by_topic: dict[str, list[Candidate]],
    output_path: Path,
    team_id: str,
    run_id: str,
    evidence_limit: int = 3,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        for topic in topics:
            rag_object = build_rag_object(
                topic,
                results_by_topic.get(topic.id, []),
                team_id=team_id,
                run_id=run_id,
                evidence_limit=evidence_limit,
            )
            sink.write(json.dumps(rag_object, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run runner helper tests and verify they pass**

Run:

```bash
pytest tests/test_run_baseline.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Commit runner helper work**

Run:

```bash
git add scripts/run_baseline.py tests/test_run_baseline.py
git commit -m "feat: add baseline runner helpers"
```

Expected: commit succeeds.

## Task 3: Validator Tests And Implementation

**Files:**
- Create: `scripts/validate_outputs.py`
- Test: `tests/test_validate_outputs.py`

- [ ] **Step 1: Write failing validator tests**

Create `tests/test_validate_outputs.py`:

```python
import json

from scripts.run_baseline import Topic
from scripts.validate_outputs import validate_rag_output, validate_retrieval_output


def test_validate_retrieval_output_accepts_ranked_rows(tmp_path):
    runfile = tmp_path / "r.tsv"
    runfile.write_text(
        "1 Q0 shard_1_1 1 10 run\n"
        "1 Q0 shard_1_2 2 9 run\n",
        encoding="utf-8",
    )

    assert validate_retrieval_output([Topic("1", "Title", "Narrative")], runfile) == []


def test_validate_retrieval_output_catches_bad_rank(tmp_path):
    runfile = tmp_path / "r.tsv"
    runfile.write_text("1 Q0 shard_1_1 2 10 run\n", encoding="utf-8")

    errors = validate_retrieval_output([Topic("1", "Title", "Narrative")], runfile)

    assert errors == ["topic 1: expected rank 1 but found 2"]


def test_validate_rag_output_accepts_valid_object(tmp_path):
    ragfile = tmp_path / "rag.jsonl"
    ragfile.write_text(
        json.dumps(
            {
                "metadata": {
                    "team_id": "team",
                    "run_id": "run",
                    "type": "automatic",
                    "narrative_id": "1",
                    "title": "Title",
                    "narrative": "Narrative",
                    "prompt": "Prompt",
                },
                "references": ["shard_1_1"],
                "answer": [{"text": "Evidence sentence.", "citations": [0]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert validate_rag_output([Topic("1", "Title", "Narrative")], ragfile) == []


def test_validate_rag_output_catches_bad_citation(tmp_path):
    ragfile = tmp_path / "rag.jsonl"
    ragfile.write_text(
        json.dumps(
            {
                "metadata": {
                    "team_id": "team",
                    "run_id": "run",
                    "type": "automatic",
                    "narrative_id": "1",
                    "title": "Title",
                    "narrative": "Narrative",
                    "prompt": "Prompt",
                },
                "references": ["shard_1_1"],
                "answer": [{"text": "Evidence sentence.", "citations": [1]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    errors = validate_rag_output([Topic("1", "Title", "Narrative")], ragfile)

    assert errors == ["topic 1 answer 1: citation 1 outside references range 0..0"]


def test_validate_rag_output_allows_insufficient_evidence_without_references(tmp_path):
    ragfile = tmp_path / "rag.jsonl"
    ragfile.write_text(
        json.dumps(
            {
                "metadata": {
                    "team_id": "team",
                    "run_id": "run",
                    "type": "automatic",
                    "narrative_id": "1",
                    "title": "Title",
                    "narrative": "Narrative",
                    "prompt": "Prompt",
                },
                "references": [],
                "answer": [
                    {
                        "text": "The retrieved evidence did not provide enough usable text to answer this topic.",
                        "citations": [],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert validate_rag_output([Topic("1", "Title", "Narrative")], ragfile) == []
```

- [ ] **Step 2: Run validator tests and verify they fail**

Run:

```bash
pytest tests/test_validate_outputs.py -q
```

Expected: tests fail with missing `scripts.validate_outputs`.

- [ ] **Step 3: Add validator implementation**

Create `scripts/validate_outputs.py`:

```python
#!/usr/bin/env python3
"""Validate TREC RAG 2026 baseline retrieval and RAG outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.run_baseline import Topic, load_topics


def validate_retrieval_output(topics: list[Topic], runfile: Path) -> list[str]:
    errors: list[str] = []
    expected_topic_ids = [topic.id for topic in topics]
    rows_by_topic: dict[str, list[tuple[int, float, int]]] = defaultdict(list)

    if not runfile.exists():
        return [f"{runfile}: file does not exist"]

    for line_number, raw_line in enumerate(runfile.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        columns = raw_line.split()
        if len(columns) != 6:
            errors.append(f"line {line_number}: expected 6 columns but found {len(columns)}")
            continue
        topic_id, q0, docid, rank_text, score_text, run_id = columns
        if q0 != "Q0":
            errors.append(f"line {line_number}: column 2 must be Q0")
        if not docid:
            errors.append(f"line {line_number}: docid is empty")
        if not run_id:
            errors.append(f"line {line_number}: run_id is empty")
        try:
            rank = int(rank_text)
        except ValueError:
            errors.append(f"line {line_number}: rank is not an integer")
            continue
        try:
            score = float(score_text)
        except ValueError:
            errors.append(f"line {line_number}: score is not numeric")
            continue
        rows_by_topic[topic_id].append((rank, score, line_number))

    for topic_id in expected_topic_ids:
        rows = rows_by_topic.get(topic_id, [])
        if not rows:
            errors.append(f"topic {topic_id}: missing retrieval rows")
            continue
        previous_score: float | None = None
        for expected_rank, (rank, score, _line_number) in enumerate(rows, start=1):
            if rank != expected_rank:
                errors.append(f"topic {topic_id}: expected rank {expected_rank} but found {rank}")
                break
            if previous_score is not None and score > previous_score:
                errors.append(f"topic {topic_id}: scores increase at rank {rank}")
                break
            previous_score = score

    unexpected_topic_ids = sorted(set(rows_by_topic) - set(expected_topic_ids))
    for topic_id in unexpected_topic_ids:
        errors.append(f"topic {topic_id}: not present in topic input")

    return errors


def _load_rag_objects(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"{path}: file does not exist"]
    objects: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            errors.append(f"line {line_number}: invalid JSON")
            continue
        if not isinstance(record, dict):
            errors.append(f"line {line_number}: expected JSON object")
            continue
        objects.append(record)
    return objects, errors


def validate_rag_output(topics: list[Topic], ragfile: Path) -> list[str]:
    objects, errors = _load_rag_objects(ragfile)
    topics_by_id = {topic.id: topic for topic in topics}
    seen_topic_ids: set[str] = set()

    for index, record in enumerate(objects, start=1):
        metadata = record.get("metadata")
        references = record.get("references")
        answer = record.get("answer")
        if not isinstance(metadata, dict):
            errors.append(f"object {index}: metadata must be an object")
            continue
        topic_id = str(metadata.get("narrative_id", ""))
        if topic_id in seen_topic_ids:
            errors.append(f"topic {topic_id}: duplicate RAG object")
        seen_topic_ids.add(topic_id)
        topic = topics_by_id.get(topic_id)
        if topic is None:
            errors.append(f"topic {topic_id}: not present in topic input")
        else:
            if metadata.get("title") != topic.title:
                errors.append(f"topic {topic_id}: metadata.title does not match topic input")
            if metadata.get("narrative") != topic.narrative:
                errors.append(f"topic {topic_id}: metadata.narrative does not match topic input")
        for field in ("team_id", "run_id", "type", "narrative_id", "title", "narrative"):
            if not metadata.get(field):
                errors.append(f"topic {topic_id or index}: metadata.{field} is required")
        if metadata.get("type") not in {"automatic", "manual"}:
            errors.append(f"topic {topic_id}: metadata.type must be automatic or manual")
        if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
            errors.append(f"topic {topic_id}: references must be a list of strings")
            references = []
        if not isinstance(answer, list) or not answer:
            errors.append(f"topic {topic_id}: answer must be a non-empty list")
            answer = []
        cited_reference_indices: set[int] = set()
        for answer_index, answer_item in enumerate(answer, start=1):
            if not isinstance(answer_item, dict):
                errors.append(f"topic {topic_id} answer {answer_index}: answer item must be an object")
                continue
            if not isinstance(answer_item.get("text"), str) or not answer_item["text"].strip():
                errors.append(f"topic {topic_id} answer {answer_index}: text is required")
            citations = answer_item.get("citations")
            if not isinstance(citations, list):
                errors.append(f"topic {topic_id} answer {answer_index}: citations must be a list")
                continue
            for citation in citations:
                if not isinstance(citation, int):
                    errors.append(f"topic {topic_id} answer {answer_index}: citation must be an integer")
                    continue
                if citation < 0 or citation >= len(references):
                    max_index = len(references) - 1
                    errors.append(
                        f"topic {topic_id} answer {answer_index}: citation {citation} "
                        f"outside references range 0..{max_index}"
                    )
                    continue
                cited_reference_indices.add(citation)
        uncited = sorted(set(range(len(references))) - cited_reference_indices)
        if uncited:
            errors.append(f"topic {topic_id}: uncited reference indices {uncited}")

    for topic_id in topics_by_id:
        if topic_id not in seen_topic_ids:
            errors.append(f"topic {topic_id}: missing RAG object")

    return errors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate TREC RAG baseline outputs.")
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--rag", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    topics = load_topics(args.topics)
    errors = []
    errors.extend(validate_retrieval_output(topics, args.retrieval))
    errors.extend(validate_rag_output(topics, args.rag))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run validator tests and verify they pass**

Run:

```bash
pytest tests/test_validate_outputs.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit validator work**

Run:

```bash
git add scripts/validate_outputs.py tests/test_validate_outputs.py
git commit -m "feat: add baseline output validators"
```

Expected: commit succeeds.

## Task 4: Pyserini REST Client And Baseline CLI

**Files:**
- Modify: `scripts/run_baseline.py`
- Test: `tests/test_run_baseline.py`

- [ ] **Step 1: Add failing tests for cache reuse and fake client execution**

Append to `tests/test_run_baseline.py`:

```python
from scripts.run_baseline import run_baseline


class FakeClient:
    def search(self, query, hits):
        assert query == "Topic"
        assert hits == 2
        return {
            "candidates": [
                {
                    "docid": "shard_1_1",
                    "rank": 1,
                    "score": 10.0,
                    "doc": {"contents": "Alpha evidence sentence."},
                },
                {
                    "docid": "shard_1_2",
                    "rank": 2,
                    "score": 9.0,
                    "doc": {"contents": "Beta evidence sentence."},
                },
            ]
        }


def test_run_baseline_writes_outputs_and_cache(tmp_path):
    topics_path = tmp_path / "topics.jsonl"
    topics_path.write_text(
        json.dumps({"id": "7", "title": "Topic", "narrative": "Narrative"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"

    run_baseline(
        topics_path=topics_path,
        output_dir=output_dir,
        client=FakeClient(),
        hits=2,
        team_id="team",
        run_id="run",
        evidence_limit=2,
        reuse_cache=False,
    )

    assert (output_dir / "cache" / "7.json").exists()
    assert (output_dir / "r_output_trec_rag_2026.tsv").read_text(encoding="utf-8") == (
        "7 Q0 shard_1_1 1 10 run\n"
        "7 Q0 shard_1_2 2 9 run\n"
    )
    rag = json.loads((output_dir / "rag_output_trec_rag_2026.jsonl").read_text(encoding="utf-8"))
    assert rag["references"] == ["shard_1_1", "shard_1_2"]


def test_run_baseline_reuses_cache_without_client_call(tmp_path):
    topics_path = tmp_path / "topics.jsonl"
    topics_path.write_text(
        json.dumps({"id": "7", "title": "Topic", "narrative": "Narrative"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "7.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "docid": "shard_1_1",
                        "rank": 1,
                        "score": 10.0,
                        "doc": {"contents": "Cached evidence sentence."},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run_baseline(
        topics_path=topics_path,
        output_dir=output_dir,
        client=None,
        hits=1,
        team_id="team",
        run_id="run",
        evidence_limit=1,
        reuse_cache=True,
    )

    assert (output_dir / "r_output_trec_rag_2026.tsv").read_text(encoding="utf-8") == (
        "7 Q0 shard_1_1 1 10 run\n"
    )
```

- [ ] **Step 2: Run the new runner tests and verify they fail**

Run:

```bash
pytest tests/test_run_baseline.py -q
```

Expected: tests fail because `run_baseline` does not exist yet.

- [ ] **Step 3: Add the REST client, cache helpers, run function, and CLI**

Append these functions and classes to `scripts/run_baseline.py` before `if __name__ == "__main__"`; if the file has no CLI block yet, add this block at the end:

```python
class PyseriniClient:
    def __init__(self, base_url: str, index: str, token: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.index = index
        self.token = token
        self.timeout = timeout

    def search(self, query: str, hits: int) -> dict[str, Any]:
        params = urllib.parse.urlencode({"query": query, "hits": str(hits)})
        url = f"{self.base_url}/v1/{urllib.parse.quote(self.index)}/search?{params}"
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Pyserini search failed with HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Pyserini search failed: {exc.reason}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Pyserini search returned non-JSON response") from exc


def _safe_cache_name(topic_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", topic_id).strip("._")
    return f"{safe or 'topic'}.json"


def _load_cached_response(cache_path: Path) -> dict[str, Any]:
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _write_cached_response(cache_path: Path, response: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_baseline(
    topics_path: Path,
    output_dir: Path,
    client: Any,
    hits: int,
    team_id: str,
    run_id: str,
    evidence_limit: int,
    reuse_cache: bool,
) -> None:
    topics = load_topics(topics_path)
    cache_dir = output_dir / "cache"
    results_by_topic: dict[str, list[Candidate]] = {}

    for topic in topics:
        cache_path = cache_dir / _safe_cache_name(topic.id)
        if reuse_cache and cache_path.exists():
            response = _load_cached_response(cache_path)
        else:
            if client is None:
                raise RuntimeError(f"topic {topic.id}: no client available and cache is missing")
            response = client.search(topic.title, hits)
            _write_cached_response(cache_path, response)
        candidates = normalize_candidates(response)
        if not candidates:
            raise RuntimeError(f"topic {topic.id}: search returned no usable candidates")
        results_by_topic[topic.id] = candidates

    write_retrieval_run(
        results_by_topic,
        output_dir / "r_output_trec_rag_2026.tsv",
        run_id=run_id,
    )
    write_rag_output(
        topics,
        results_by_topic,
        output_dir / "rag_output_trec_rag_2026.jsonl",
        team_id=team_id,
        run_id=run_id,
        evidence_limit=evidence_limit,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TREC RAG 2026 Pyserini baseline.")
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--hits", type=int, default=DEFAULT_HITS)
    parser.add_argument("--evidence-limit", type=int, default=3)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--team-id", default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser


def main() -> int:
    load_dotenv(Path(".env"))
    load_dotenv(Path(".env.local"))
    args = build_arg_parser().parse_args()
    base_url = args.base_url or os.environ.get("INDEX_URL") or DEFAULT_BASE_URL
    team_id = args.team_id or os.environ.get("TREC_RAG_TEAM_ID") or DEFAULT_TEAM_ID
    token = os.environ.get("PYSERINI_API_TOKEN")
    client = None
    if not args.reuse_cache:
        if not token:
            print("ERROR: PYSERINI_API_TOKEN is required unless --reuse-cache can satisfy every topic")
            return 1
        client = PyseriniClient(base_url=base_url, index=args.index, token=token, timeout=args.timeout)
    try:
        run_baseline(
            topics_path=args.topics,
            output_dir=args.output_dir,
            client=client,
            hits=args.hits,
            team_id=team_id,
            run_id=args.run_id,
            evidence_limit=args.evidence_limit,
            reuse_cache=args.reuse_cache,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run runner tests and verify they pass**

Run:

```bash
pytest tests/test_run_baseline.py -q
```

Expected: all runner tests pass.

- [ ] **Step 5: Run all unit tests**

Run:

```bash
pytest -q
```

Expected: converter, runner, and validator tests pass.

- [ ] **Step 6: Commit CLI runner work**

Run:

```bash
git add scripts/run_baseline.py tests/test_run_baseline.py
git commit -m "feat: add Pyserini baseline runner CLI"
```

Expected: commit succeeds.

## Task 5: Ignore Rules And README

**Files:**
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Add ignore rules for secrets and generated files**

Create `.gitignore`:

```gitignore
.env
.env.local
.curlrc.pyserini-rest
__pycache__/
.pytest_cache/
*.pyc
outputs/
tmp/
```

- [ ] **Step 2: Add README commands**

Create `README.md`:

````markdown
# TREC RAG Baseline

This workspace contains a small baseline for TREC RAG 2026. The baseline uses Pyserini BM25 over the ClimbMix index (`climbmix-400b`) and writes both Retrieval and RAG outputs.

## Convert Development Topics

```bash
python -m scripts.convert_dev_topics \
  trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv \
  outputs/baseline/topics/rag25-topics-dev.jsonl
```

## Run Baseline

The Pyserini API token must be available as `PYSERINI_API_TOKEN` in the environment, `.env`, or `.env.local`.

```bash
python -m scripts.run_baseline \
  --topics outputs/baseline/topics/rag25-topics-dev.jsonl \
  --output-dir outputs/baseline \
  --hits 100
```

Generated files:

- `outputs/baseline/r_output_trec_rag_2026.tsv`
- `outputs/baseline/rag_output_trec_rag_2026.jsonl`
- `outputs/baseline/cache/*.json`

## Validate Outputs

```bash
python -m scripts.validate_outputs \
  --topics outputs/baseline/topics/rag25-topics-dev.jsonl \
  --retrieval outputs/baseline/r_output_trec_rag_2026.tsv \
  --rag outputs/baseline/rag_output_trec_rag_2026.jsonl
```

## Run Tests

```bash
pytest -q
```

## Notes

The first RAG baseline is deterministic and extractive. It emits short cited evidence sentences from retrieved document text and marks the run as `metadata.type: "automatic"`.
````

- [ ] **Step 3: Commit docs and ignore rules**

Run:

```bash
git add .gitignore README.md
git commit -m "docs: add baseline usage instructions"
```

Expected: commit succeeds.

## Task 6: Local Verification And One-Topic Smoke Test

**Files:**
- Generated only: `outputs/baseline/...`

- [ ] **Step 1: Run the full unit test suite**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Convert development topics**

Run:

```bash
python -m scripts.convert_dev_topics \
  trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv \
  outputs/baseline/topics/rag25-topics-dev.jsonl
```

Expected: command prints `Wrote N topics to outputs/baseline/topics/rag25-topics-dev.jsonl`, where `N` is greater than zero.

- [ ] **Step 3: Create a one-topic JSONL smoke input**

Run:

```bash
head -n 1 outputs/baseline/topics/rag25-topics-dev.jsonl > outputs/baseline/topics/smoke-one-topic.jsonl
```

Expected: `outputs/baseline/topics/smoke-one-topic.jsonl` contains exactly one JSONL topic.

- [ ] **Step 4: Run the live Pyserini smoke test**

Run:

```bash
python -m scripts.run_baseline \
  --topics outputs/baseline/topics/smoke-one-topic.jsonl \
  --output-dir outputs/baseline/smoke \
  --hits 5 \
  --evidence-limit 2
```

Expected: command prints `Wrote outputs to outputs/baseline/smoke`. If it prints `ERROR: PYSERINI_API_TOKEN is required`, confirm the token is available locally without printing it.

- [ ] **Step 5: Validate smoke outputs**

Run:

```bash
python -m scripts.validate_outputs \
  --topics outputs/baseline/topics/smoke-one-topic.jsonl \
  --retrieval outputs/baseline/smoke/r_output_trec_rag_2026.tsv \
  --rag outputs/baseline/smoke/rag_output_trec_rag_2026.jsonl
```

Expected: `Validation passed`.

- [ ] **Step 6: Inspect output shapes without printing secrets**

Run:

```bash
wc -l outputs/baseline/smoke/r_output_trec_rag_2026.tsv outputs/baseline/smoke/rag_output_trec_rag_2026.jsonl
python -m json.tool outputs/baseline/smoke/rag_output_trec_rag_2026.jsonl | sed -n '1,80p'
```

Expected: retrieval output has 5 lines and RAG output has 1 line. The pretty-printed RAG object has `metadata`, `references`, and `answer`.

- [ ] **Step 7: Report final status**

Report:

```text
Implemented baseline scripts, tests, README, and ignore rules.
Verification:
- pytest -q: PASS
- dev topic conversion: PASS
- one-topic live Pyserini smoke test: PASS or blocked with token/API reason
- smoke validation: PASS if smoke test ran
```

Do not commit generated files under `outputs/`.
