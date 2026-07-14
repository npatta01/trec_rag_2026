"""Plan and execute local, query-isolated MiniLM window scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .facet_local_minilm_preflight import (
    MAX_WINDOWS_PER_DOCUMENT,
    MIN_PASSAGE_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PAIR_MAX_TOKENS,
    PASSAGE_OVERLAP_TOKENS,
    QUERY_MAX_TOKENS,
    WINDOW_POLICY_VERSION,
    WindowPlanRow,
    build_window_plan,
    load_verified_materialization,
    load_verified_tokenizer,
    score_cache_context,
)
from .repo_env import find_repo_root, repo_cache_root
from .rerank_score_cache import GlobalScoreCache


PREFLIGHT_SCHEMA_VERSION = "adaptive-evidence-score-preflight-v1"
BASE_SCORE_SCHEMA_VERSION = "adaptive-evidence-window-score-v1"
BASE_SHARD_SCHEMA_VERSION = "adaptive-evidence-score-shard-v1"
BASE_RECEIPT_SCHEMA_VERSION = "adaptive-evidence-score-base-v1"
CONTRACT_SCHEMA_VERSION = "adaptive-evidence-contract-v1"
EXPECTED_DOCUMENT_COUNT = 8_114
EXPECTED_BROAD_COUNT = 4
EXPECTED_O0_COUNT = 24
REFERENCE_PAIRS_PER_SECOND = 300.0
REFERENCE_FIXED_SECONDS = 30.0
SCORING_BATCH_SIZE = 32
DEFAULT_MODEL_RECEIPT = Path(
    "outputs/rag25_facet_local_minilm_v1/model_v1/materialization.json"
)


def _generated_by(document: Mapping[str, object], facet_id: str) -> bool:
    return any(
        row.get("family") == "facet" and row.get("facet_id") == facet_id
        for row in document["provenance"]  # type: ignore[index]
    )


def _reject_protected(topic_ids: Sequence[object]) -> None:
    for topic_id in map(str, topic_ids):
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")


def _require_pilot_topic_ids(topic_ids: Sequence[object]) -> None:
    allowed = set(PILOT_TOPIC_IDS)
    for topic_id in topic_ids:
        if not isinstance(topic_id, str) or topic_id not in allowed:
            raise ValueError(
                "scoring topic_id must be an exact frozen pilot topic "
                f"from {','.join(PILOT_TOPIC_IDS)}"
            )


def _validate_contract_population(
    obligations: Sequence[Mapping[str, object]],
    documents: Sequence[Mapping[str, object]],
    *,
    expected_document_count: int | None = None,
    expected_topic_ids: Sequence[str] | None = None,
) -> None:
    identities = {
        (str(row.get("topic_id")), str(row.get("document_id")))
        for row in documents
    }
    required_count = expected_document_count or len(documents)
    if len(identities) != required_count or len(documents) != required_count:
        raise ValueError(
            f"contract requires exactly {required_count} unique topic-document identities"
        )
    required_topics = set(
        expected_topic_ids
        if expected_topic_ids is not None
        else (str(row.get("topic_id")) for row in documents)
    )
    broad_counts = {
        topic_id: sum(
            row.get("kind") == "broad" and str(row.get("topic_id")) == topic_id
            for row in obligations
        )
        for topic_id in required_topics
    }
    broad_topics = {
        str(row.get("topic_id")) for row in obligations if row.get("kind") == "broad"
    }
    if broad_topics != required_topics or any(count != 1 for count in broad_counts.values()):
        raise ValueError("contract requires exactly one broad obligation per pilot topic")


def _validated_obligations(
    obligations: Sequence[Mapping[str, object]],
) -> tuple[dict[str, Mapping[str, object]], dict[str, Mapping[str, object]]]:
    by_id: dict[str, Mapping[str, object]] = {}
    for row in obligations:
        obligation_id = str(row.get("obligation_id"))
        if obligation_id in by_id:
            raise ValueError(f"duplicate obligation ID {obligation_id}")
        by_id[obligation_id] = row
    o1_parents: dict[str, Mapping[str, object]] = {}
    for row in obligations:
        if row.get("kind") != "o1":
            continue
        obligation_id = str(row.get("obligation_id"))
        parent_id = str(row.get("parent_id"))
        parent = by_id.get(parent_id)
        if parent is None:
            raise ValueError(f"O1 parent {parent_id} does not exist")
        if parent.get("kind") != "o0":
            raise ValueError(f"O1 parent {parent_id} must be kind o0")
        topic_id = str(row.get("topic_id"))
        if str(parent.get("topic_id")) != topic_id:
            raise ValueError(f"O1 parent {parent_id} must share topic {topic_id}")
        source_facet_id = parent.get("source_facet_id")
        if (
            not isinstance(source_facet_id, str)
            or not source_facet_id
            or source_facet_id != parent_id
            or str(parent.get("obligation_id")) != parent_id
        ):
            raise ValueError(f"O1 parent {parent_id} source facet is invalid")
        o1_parents[obligation_id] = parent
    return by_id, o1_parents


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_compact_json_bytes(dict(row)) for row in rows)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    try:
        source = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line:
            raise ValueError(f"{label}:{line_number} is blank")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number} must be a JSON object")
        if _compact_json_bytes(row).rstrip(b"\n") != line:
            raise ValueError(f"{label}:{line_number} is not canonical JSON")
        rows.append(row)
    return rows


def load_score_contract(
    contract_dir: Path,
    *,
    expected_document_count: int = EXPECTED_DOCUMENT_COUNT,
    expected_o0_count: int = EXPECTED_O0_COUNT,
    expected_broad_count: int = EXPECTED_BROAD_COUNT,
) -> dict[str, object]:
    """Load and authenticate the Task 1 contract before model/cache access."""

    root = Path(contract_dir)
    manifest = _read_json(root / "manifest.json", "contract manifest")
    raw_topic_ids = manifest.get("topic_ids")
    if not isinstance(raw_topic_ids, list):
        raise ValueError("contract manifest topic_ids must be an array")
    topic_ids = [str(value) for value in raw_topic_ids]
    _reject_protected(topic_ids)
    if len(topic_ids) != len(PILOT_TOPIC_IDS) or set(topic_ids) != set(
        PILOT_TOPIC_IDS
    ):
        raise ValueError("contract must contain exactly the four pilot topics")
    if (
        manifest.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or manifest.get("qrels_opened") is not False
        or manifest.get("document_count") != expected_document_count
        or manifest.get("broad_obligation_count") != expected_broad_count
        or manifest.get("o0_obligation_count") != expected_o0_count
    ):
        raise ValueError("contract manifest counts or safety binding differ")

    summary = _read_json(root / "summary.json", "contract summary")
    if (
        summary.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or summary.get("status") != "complete"
        or summary.get("topic_ids") != raw_topic_ids
        or summary.get("qrels_opened") is not False
        or summary.get("protected_topic_count") != 0
        or summary.get("document_count") != expected_document_count
        or summary.get("broad_obligation_count") != expected_broad_count
        or summary.get("o0_obligation_count") != expected_o0_count
    ):
        raise ValueError("contract summary counts or safety binding differ")
    raw_hashes = summary.get("artifact_sha256")
    if not isinstance(raw_hashes, Mapping):
        raise ValueError("contract summary artifact hashes are missing")
    artifact_names = (
        "manifest.json",
        "obligations.jsonl",
        "documents.jsonl",
        "folds.jsonl",
    )
    for name in artifact_names:
        expected = raw_hashes.get(name)
        try:
            observed = _sha256_file(root / name)
        except OSError as exc:
            raise ValueError(f"contract {name} is unreadable") from exc
        if expected != observed:
            raise ValueError(f"contract {name} hash differs")

    obligations = _read_jsonl(root / "obligations.jsonl", "contract obligations")
    documents = _read_jsonl(root / "documents.jsonl", "contract documents")
    _reject_protected(
        [row.get("topic_id") for row in obligations]
        + [row.get("topic_id") for row in documents]
    )
    if (
        len(documents) != expected_document_count
        or sum(row.get("kind") == "broad" for row in obligations)
        != expected_broad_count
        or sum(row.get("kind") == "o0" for row in obligations) != expected_o0_count
    ):
        raise ValueError("contract row counts differ from its receipt")
    _validate_contract_population(
        obligations,
        documents,
        expected_document_count=expected_document_count,
        expected_topic_ids=PILOT_TOPIC_IDS,
    )
    if {str(row.get("topic_id")) for row in obligations} != set(PILOT_TOPIC_IDS):
        raise ValueError("contract obligation topics differ from the pilot set")
    if {str(row.get("topic_id")) for row in documents} != set(PILOT_TOPIC_IDS):
        raise ValueError("contract document topics differ from the pilot set")
    for row in documents:
        text = row.get("text")
        if not isinstance(text, str) or row.get("text_sha256") != _sha256_bytes(
            text.encode("utf-8")
        ):
            raise ValueError("contract document text hash differs")
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "obligations": obligations,
        "documents": documents,
    }


def build_score_candidates(
    contract: Mapping[str, object],
    *,
    derived: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Build broad-universal, O0 facet-local, and O1 parent-local rows."""

    contract_obligations = list(contract["obligations"])  # type: ignore[index]
    documents = list(contract["documents"])  # type: ignore[index]
    _reject_protected(
        [row["topic_id"] for row in contract_obligations]
        + [row["topic_id"] for row in derived]
        + [row["topic_id"] for row in documents]
    )
    _validate_contract_population(contract_obligations, documents)
    obligations = [*contract_obligations, *derived]
    _by_id, o1_parents = _validated_obligations(obligations)
    output: list[dict[str, object]] = []
    for document in documents:
        topic_id = str(document["topic_id"])
        for obligation in obligations:
            if str(obligation["topic_id"]) != topic_id:
                continue
            kind = str(obligation["kind"])
            if kind == "o0" and not _generated_by(
                document, str(obligation["source_facet_id"])
            ):
                continue
            if kind == "o1":
                parent = o1_parents[str(obligation["obligation_id"])]
                if not _generated_by(document, str(parent["source_facet_id"])):
                    continue
            query = str(obligation["query"])
            text = str(document["text"])
            output.append(
                {
                    "topic_id": topic_id,
                    "obligation_id": str(obligation["obligation_id"]),
                    "family": kind,
                    "variant": str(obligation["obligation_id"]),
                    "rank": int(document["union_order"]),
                    "document_id": str(document["document_id"]),
                    "query": query,
                    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "fold": int(document["fold"]),
                }
            )
    return sorted(
        output,
        key=lambda row: (row["topic_id"], row["obligation_id"], row["rank"]),
    )


def _coverage_row(
    *,
    document_count: int,
    candidate_count: int,
    windows: Sequence[WindowPlanRow],
) -> dict[str, object]:
    keys = {row.cache_key for row in windows}
    hit_keys = {row.cache_key for row in windows if row.cache_hit}
    return {
        "document_count": document_count,
        "candidate_count": candidate_count,
        "window_count": len(windows),
        "unique_pair_count": len(keys),
        "cache_hit_count": len(hit_keys),
        "cache_miss_count": len(keys - hit_keys),
        "cache_hit_window_count": sum(row.cache_hit for row in windows),
        "cache_miss_window_count": sum(not row.cache_hit for row in windows),
    }


def build_score_preflight(
    candidates: Sequence[Mapping[str, object]],
    tokenizer: object,
    cache_lookup: Callable[[str, str], float | None],
) -> dict[str, object]:
    """Build complete candidate/window/cache coverage without inference."""

    materialized = tuple(candidates)
    _reject_protected([row.get("topic_id") for row in materialized])
    if not materialized:
        raise ValueError("score preflight requires at least one candidate")
    windows: list[WindowPlanRow] = []
    cache_results: dict[str, bool] = {}
    for candidate in materialized:
        query = str(candidate["query"])
        candidate_windows = build_window_plan(candidate, tokenizer, query=query)
        for row in candidate_windows:
            if row.cache_key not in cache_results:
                cache_results[row.cache_key] = (
                    cache_lookup(row.query, row.window_text) is not None
                )
            hit = cache_results[row.cache_key]
            windows.append(replace(row, cache_hit=hit))

    by_obligation: dict[tuple[str, str], list[WindowPlanRow]] = defaultdict(list)
    by_document: dict[tuple[str, str, str], list[WindowPlanRow]] = defaultdict(list)
    by_topic: dict[str, list[WindowPlanRow]] = defaultdict(list)
    for row in windows:
        by_obligation[(row.topic_id, row.variant)].append(row)
        by_document[(row.topic_id, row.variant, row.document_id)].append(row)
        by_topic[row.topic_id].append(row)
    obligation_candidates: dict[tuple[str, str], int] = defaultdict(int)
    topic_candidates: dict[str, int] = defaultdict(int)
    obligation_documents: dict[tuple[str, str], set[str]] = defaultdict(set)
    topic_documents: dict[str, set[str]] = defaultdict(set)
    all_documents: set[tuple[str, str]] = set()
    for row in materialized:
        topic_id = str(row["topic_id"])
        obligation_id = str(row["obligation_id"])
        document_id = str(row["document_id"])
        obligation_candidates[(topic_id, obligation_id)] += 1
        topic_candidates[topic_id] += 1
        obligation_documents[(topic_id, obligation_id)].add(document_id)
        topic_documents[topic_id].add(document_id)
        all_documents.add((topic_id, document_id))

    return {
        "qrels_opened": False,
        "inference_count": 0,
        "summary": _coverage_row(
            document_count=len(all_documents),
            candidate_count=len(materialized),
            windows=windows,
        ),
        "topics": [
            {
                "topic_id": topic_id,
                **_coverage_row(
                    document_count=len(topic_documents[topic_id]),
                    candidate_count=topic_candidates[topic_id],
                    windows=by_topic[topic_id],
                ),
            }
            for topic_id in sorted(by_topic)
        ],
        "obligations": [
            {
                "topic_id": topic_id,
                "obligation_id": obligation_id,
                **_coverage_row(
                    document_count=len(
                        obligation_documents[(topic_id, obligation_id)]
                    ),
                    candidate_count=obligation_candidates[(topic_id, obligation_id)],
                    windows=by_obligation[(topic_id, obligation_id)],
                ),
            }
            for topic_id, obligation_id in sorted(by_obligation)
        ],
        "documents": [
            {
                "topic_id": topic_id,
                "obligation_id": obligation_id,
                "document_id": document_id,
                "document_token_coverage_fraction": rows[0].document_token_coverage_fraction,
                **_coverage_row(document_count=1, candidate_count=1, windows=rows),
            }
            for (topic_id, obligation_id, document_id), rows in sorted(
                by_document.items()
            )
        ],
        "windows": [row.to_dict() for row in windows],
    }


def score_window_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    predict: Callable[[list[tuple[str, str]]], Sequence[float]],
    cache_get: Callable[[str, str], float | None],
    cache_add: Callable[[Sequence[tuple[str, str, float]]], object],
) -> list[dict[str, object]]:
    """Score unique injected misses and restore every frozen window row."""

    for row in rows:
        topic_id = row.get("topic_id")
        if not isinstance(topic_id, str) or not topic_id:
            raise ValueError("window topic_id must be nonempty text")
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    unique: dict[tuple[str, str], None] = {}
    initial_scores: dict[tuple[str, str], float | None] = {}
    for row in rows:
        pair = (str(row["query"]), str(row["window_text"]))
        unique.setdefault(pair, None)
    for pair in unique:
        value = cache_get(*pair)
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("score cache contains a nonfinite score")
        initial_scores[pair] = value

    missing = sorted(pair for pair, value in initial_scores.items() if value is None)
    if missing:
        values = list(predict(missing))
        if len(values) != len(missing):
            raise ValueError("predictor score count differs from missing pairs")
        additions: list[tuple[str, str, float]] = []
        for (query, text), score in zip(missing, values, strict=True):
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("predictor returned a nonfinite score")
            additions.append((query, text, value))
        cache_add(additions)

    output: list[dict[str, object]] = []
    for row in rows:
        query, text = str(row["query"]), str(row["window_text"])
        score = cache_get(query, text)
        if score is None:
            raise ValueError("score cache does not cover a frozen window")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("score cache contains a nonfinite score")
        output.append(
            {
                **dict(row),
                "score": value,
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
    return output


def verify_completed_shard(path: Path, receipt: Mapping[str, object]) -> int:
    """Verify one create-only score shard against its completion receipt."""

    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"completed shard is unreadable: {path}") from exc
    if _sha256_bytes(payload) != receipt.get("sha256"):
        raise ValueError("completed shard hash differs from its receipt")
    rows = sum(1 for line in payload.splitlines() if line)
    expected_rows = receipt.get("rows")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
        raise ValueError("completed shard receipt row count is invalid")
    if rows != expected_rows:
        raise ValueError("completed shard row count differs from its receipt")
    return rows


def _exclusive_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as sink:
        sink.write(value)
        sink.flush()
        os.fsync(sink.fileno())


def _exclusive_json(path: Path, value: Mapping[str, object]) -> None:
    _exclusive_bytes(path, _pretty_json_bytes(value))


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_possible_forward_bookkeeping(
    forward_pairs: int,
    forward_calls: int,
    label: str,
) -> None:
    if (forward_pairs == 0) != (forward_calls == 0):
        raise ValueError(f"{label} forward execution bookkeeping is impossible")


def _artifact_binding(
    root: Path,
    name: str,
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    raw = artifacts.get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"preflight {name} artifact binding is missing")
    path = root / name
    try:
        size = path.stat().st_size
        sha256 = _sha256_file(path)
    except OSError as exc:
        raise ValueError(f"preflight {name} is unreadable") from exc
    rows = _require_nonnegative_int(raw.get("rows"), f"preflight {name} rows")
    expected_bytes = _require_nonnegative_int(
        raw.get("bytes"), f"preflight {name} bytes"
    )
    if size != expected_bytes:
        raise ValueError(f"preflight {name} byte count differs")
    if sha256 != raw.get("sha256"):
        raise ValueError(f"preflight {name} hash differs")
    return {"rows": rows, "bytes": size, "sha256": sha256}


def _preflight_topics(receipt: Mapping[str, object]) -> list[str]:
    coverage = receipt.get("coverage_matrix")
    if not isinstance(coverage, Mapping):
        raise ValueError("score preflight coverage matrix is missing")
    raw_topics = coverage.get("topics")
    raw_obligations = coverage.get("obligations")
    if not isinstance(raw_topics, list) or not isinstance(raw_obligations, list):
        raise ValueError("score preflight topic or obligation coverage is missing")
    topics: list[str] = []
    for row in [*raw_topics, *raw_obligations]:
        if not isinstance(row, Mapping):
            raise ValueError("score preflight coverage rows must be objects")
        topic_id = row.get("topic_id")
        if not isinstance(topic_id, str) or not topic_id:
            raise ValueError("score preflight topic IDs must be nonempty text")
        topics.append(topic_id)
    _reject_protected(topics)
    _require_pilot_topic_ids(topics)
    return topics


def _load_scoring_preflight(
    preflight_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    root = Path(preflight_dir)
    receipt_path = root / "preflight.json"
    receipt = _read_json(receipt_path, "score preflight receipt")
    if (
        receipt.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or receipt.get("status") != "tokenizer_only_preflight_complete"
        or receipt.get("qrels_opened") is not False
        or receipt.get("network") is not False
        or receipt.get("external_cost_usd") != 0.0
        or receipt.get("inference_count") != 0
        or receipt.get("model_constructed") is not False
        or receipt.get("model") != MODEL_ID
        or receipt.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("score preflight differs from the frozen local scorer contract")

    # The receipt exposes every topic queue, so protected topics fail before any
    # large source artifact, score cache, or model snapshot is opened.
    _preflight_topics(receipt)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("score preflight artifact bindings are missing")
    candidate_binding = _artifact_binding(root, "candidates.jsonl", artifacts)
    window_binding = _artifact_binding(root, "windows.jsonl", artifacts)
    windows = _read_jsonl(root / "windows.jsonl", "score preflight windows")
    _reject_protected([row.get("topic_id") for row in windows])
    _require_pilot_topic_ids([row.get("topic_id") for row in windows])
    if len(windows) != window_binding["rows"]:
        raise ValueError("preflight windows row count differs")
    if not windows:
        raise ValueError("score preflight must contain frozen windows")

    pair_by_key: dict[str, tuple[str, str, bool]] = {}
    obligations: dict[tuple[str, str], int] = defaultdict(int)
    for row in windows:
        topic_id = row.get("topic_id")
        obligation_id = row.get("variant")
        cache_key = row.get("cache_key")
        query = row.get("query")
        text = row.get("window_text")
        cache_hit = row.get("cache_hit")
        if (
            not isinstance(topic_id, str)
            or not topic_id
            or not isinstance(obligation_id, str)
            or not obligation_id
            or not isinstance(cache_key, str)
            or not cache_key
            or not isinstance(query, str)
            or not isinstance(text, str)
            or not isinstance(cache_hit, bool)
        ):
            raise ValueError("frozen window identity or cache provenance is invalid")
        pair = (query, text, cache_hit)
        previous = pair_by_key.setdefault(cache_key, pair)
        if previous != pair:
            raise ValueError("frozen cache key has conflicting pair provenance")
        obligations[(topic_id, obligation_id)] += 1

    summary = receipt.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("score preflight summary is missing")
    expected_summary = {
        "window_count": len(windows),
        "unique_pair_count": len(pair_by_key),
        "cache_hit_count": sum(value[2] for value in pair_by_key.values()),
        "cache_miss_count": sum(not value[2] for value in pair_by_key.values()),
        "cache_hit_window_count": sum(bool(row["cache_hit"]) for row in windows),
        "cache_miss_window_count": sum(not bool(row["cache_hit"]) for row in windows),
    }
    for name, expected in expected_summary.items():
        if summary.get(name) != expected:
            raise ValueError(f"score preflight summary {name} differs")

    coverage = receipt["coverage_matrix"]
    assert isinstance(coverage, Mapping)
    raw_obligations = coverage["obligations"]
    assert isinstance(raw_obligations, list)
    received_obligations: dict[tuple[str, str], int] = {}
    for row in raw_obligations:
        assert isinstance(row, Mapping)
        key = (str(row["topic_id"]), str(row.get("obligation_id")))
        if key in received_obligations:
            raise ValueError("score preflight has duplicate obligation coverage")
        received_obligations[key] = _require_nonnegative_int(
            row.get("window_count"), "obligation window count"
        )
    if received_obligations != dict(obligations):
        raise ValueError("score preflight obligation coverage differs from windows")
    return receipt, windows, {
        "preflight_dir": str(root.resolve()),
        "preflight_sha256": _sha256_file(receipt_path),
        "candidates": candidate_binding,
        "windows": window_binding,
    }


def _shard_filename(topic_id: str, obligation_id: str) -> str:
    _require_pilot_topic_ids([topic_id])
    local_id = obligation_id
    for separator in (":", "-"):
        prefix = f"{topic_id}{separator}"
        if local_id.startswith(prefix):
            local_id = local_id[len(prefix) :]
            break
    safe = quote(local_id, safe="-_.")
    if not safe:
        raise ValueError("obligation ID cannot form a score shard name")
    return f"{quote(topic_id, safe='')}__{safe}.jsonl"


def _confined_shard_paths(output_dir: Path, filename: str) -> tuple[Path, Path]:
    if Path(filename).name != filename:
        raise ValueError("score shard filename is not a single confined component")
    destination = Path(output_dir).resolve(strict=False)
    shard_root = destination / "shards"
    if shard_root.resolve(strict=False) != shard_root:
        raise ValueError("score shard directory is not confined beneath output")
    shard_path = (shard_root / filename).resolve(strict=False)
    sidecar_path = shard_path.with_suffix(".receipt.json").resolve(strict=False)
    if shard_path.parent != shard_root or sidecar_path.parent != shard_root:
        raise ValueError("score shard paths are not confined beneath output/shards")
    return shard_path, sidecar_path


def _planned_score_rows(
    scored: Sequence[Mapping[str, object]],
    *,
    preflight_sha256: str,
    windows_sha256: str,
    execution: Mapping[str, object],
) -> list[dict[str, object]]:
    return [
        {
            **dict(row),
            "score_schema_version": BASE_SCORE_SCHEMA_VERSION,
            "score_representation": "raw_logits",
            "inference_dtype": "float32",
            "device": execution.get("device", "cuda"),
            "execution_backend": execution.get("execution_backend", "rocm"),
            "preflight_sha256": preflight_sha256,
            "preflight_windows_sha256": windows_sha256,
        }
        for row in scored
    ]


def _validate_score_shard_rows(
    path: Path,
    planned: Sequence[Mapping[str, object]],
    *,
    preflight_sha256: str,
    windows_sha256: str,
) -> None:
    observed = _read_jsonl(path, "completed score shard")
    if len(observed) != len(planned):
        raise ValueError("completed shard row count differs from frozen windows")
    additions = {
        "score_schema_version",
        "score",
        "model",
        "model_revision",
        "score_representation",
        "inference_dtype",
        "device",
        "execution_backend",
        "preflight_sha256",
        "preflight_windows_sha256",
    }
    for source, score in zip(planned, observed, strict=True):
        if set(score) != set(source) | additions:
            raise ValueError("completed shard score fields differ")
        if any(score.get(name) != value for name, value in source.items()):
            raise ValueError("completed shard window provenance differs")
        value = score.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("completed shard score is invalid")
        if not math.isfinite(float(value)):
            raise ValueError("completed shard score is nonfinite")
        if (
            score.get("model") != MODEL_ID
            or score.get("model_revision") != MODEL_REVISION
            or score.get("score_schema_version") != BASE_SCORE_SCHEMA_VERSION
            or score.get("score_representation") != "raw_logits"
            or score.get("inference_dtype") != "float32"
            or (
                score.get("device"), score.get("execution_backend")
            )
            not in {
                ("cuda", "rocm"),
                ("cache", "global_score_cache"),
            }
            or score.get("preflight_sha256") != preflight_sha256
            or score.get("preflight_windows_sha256") != windows_sha256
        ):
            raise ValueError(
                "completed shard model, device, or preflight binding differs"
            )


def _load_complete_shard(
    shard_path: Path,
    receipt_path: Path,
    planned: Sequence[Mapping[str, object]],
    *,
    topic_id: str,
    obligation_id: str,
    preflight_sha256: str,
    windows_sha256: str,
) -> dict[str, object] | None:
    shard_exists = shard_path.exists()
    receipt_exists = receipt_path.exists()
    if not shard_exists and not receipt_exists:
        return None
    if shard_exists != receipt_exists:
        raise ValueError("incomplete create-only score shard cannot be resumed")
    receipt = _read_json(receipt_path, "score shard receipt")
    if (
        receipt.get("schema_version") != BASE_SHARD_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("topic_id") != topic_id
        or receipt.get("obligation_id") != obligation_id
        or receipt.get("path") != f"shards/{shard_path.name}"
        or receipt.get("preflight_sha256") != preflight_sha256
        or receipt.get("preflight_windows_sha256") != windows_sha256
        or receipt.get("model") != MODEL_ID
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("qrels_opened") is not False
        or receipt.get("network_call_count") != 0
        or receipt.get("paid_call_count") != 0
    ):
        raise ValueError("completed score shard receipt binding differs")
    if verify_completed_shard(shard_path, receipt) != len(planned):
        raise ValueError("completed shard coverage differs from frozen windows")
    if receipt.get("bytes") != shard_path.stat().st_size:
        raise ValueError("completed shard byte count differs from its receipt")
    _validate_score_shard_rows(
        shard_path,
        planned,
        preflight_sha256=preflight_sha256,
        windows_sha256=windows_sha256,
    )
    return receipt


def _predictor_execution(predictor: object | None) -> dict[str, object]:
    receipt = getattr(predictor, "execution_receipt", None)
    if callable(receipt):
        value = receipt()
        if isinstance(value, Mapping):
            return dict(value)
    if isinstance(receipt, Mapping):
        return dict(receipt)
    return {
        "device": "cache",
        "execution_backend": "global_score_cache",
        "device_name": "not_applicable_cache_only",
        "torch_version": "not_loaded_cache_only",
        "torch_hip_version": "not_loaded_cache_only",
        "peak_device_memory_bytes": 0,
        "peak_host_memory_bytes": 0,
    }


def _aggregate_shard_execution(
    shard_receipts: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, object] | None:
    records = [
        dict(value)
        for receipt in shard_receipts.values()
        if isinstance((value := receipt.get("execution")), Mapping)
    ]
    if not records:
        return None
    identity_fields = (
        "device",
        "execution_backend",
        "device_name",
        "torch_version",
        "torch_hip_version",
    )
    identity = {name: records[0].get(name) for name in identity_fields}
    if any(
        any(record.get(name) != expected for name, expected in identity.items())
        for record in records[1:]
    ):
        raise ValueError("resumed shard ROCm execution identity differs")
    return {
        **identity,
        "peak_device_memory_bytes": max(
            int(record.get("peak_device_memory_bytes", 0)) for record in records
        ),
        "peak_host_memory_bytes": max(
            int(record.get("peak_host_memory_bytes", 0)) for record in records
        ),
    }


def _host_memory_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class _RocmPredictor:
    def __init__(
        self,
        *,
        runtime: object,
        tokenizer: object,
        model: object,
        device_name: str,
    ) -> None:
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.model = model
        self.device_name = device_name
        self.forward_pair_count = 0
        self.forward_call_count = 0
        self.elapsed_seconds = 0.0
        self.peak_device_memory_bytes = 0
        self.peak_host_memory_bytes = int(runtime.host_memory_bytes())  # type: ignore[attr-defined]
        runtime.torch.cuda.reset_peak_memory_stats()  # type: ignore[attr-defined]

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), SCORING_BATCH_SIZE):
            batch = pairs[start : start + SCORING_BATCH_SIZE]
            started = self.runtime.clock()  # type: ignore[attr-defined]
            encoded = self.tokenizer(  # type: ignore[operator]
                [query for query, _text in batch],
                [text for _query, text in batch],
                padding=True,
                truncation=False,
                max_length=PAIR_MAX_TOKENS,
                return_tensors="pt",
            )
            if not isinstance(encoded, Mapping):
                raise ValueError("MiniLM tokenizer output must be a mapping")
            inputs = {
                name: tensor.to("cuda")  # type: ignore[attr-defined]
                for name, tensor in encoded.items()
            }
            self.runtime.torch.cuda.synchronize()  # type: ignore[attr-defined]
            try:
                with self.runtime.torch.inference_mode():  # type: ignore[attr-defined]
                    output = self.model(**inputs)  # type: ignore[operator]
            except BaseException as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        "ROCm out of memory; CPU or network fallback is forbidden"
                    ) from exc
                raise
            self.runtime.torch.cuda.synchronize()  # type: ignore[attr-defined]
            elapsed = float(self.runtime.clock() - started)  # type: ignore[attr-defined]
            if elapsed <= 0:
                raise RuntimeError("MiniLM inference clock did not advance")
            logits = getattr(output, "logits", None)
            shape = tuple(getattr(logits, "shape", ()))
            if shape != (len(batch), 1):
                raise ValueError("MiniLM logits must have shape (batch, 1)")
            values = logits.detach().float().cpu().tolist()
            if len(values) != len(batch) or any(
                not isinstance(row, list) or len(row) != 1 for row in values
            ):
                raise ValueError("MiniLM logits must have shape (batch, 1)")
            for row in values:
                value = float(row[0])
                if not math.isfinite(value):
                    raise ValueError("MiniLM predictor returned a nonfinite score")
                scores.append(value)
            self.forward_pair_count += len(batch)
            self.forward_call_count += 1
            self.elapsed_seconds += elapsed
            self.peak_device_memory_bytes = max(
                self.peak_device_memory_bytes,
                int(self.runtime.torch.cuda.max_memory_allocated()),  # type: ignore[attr-defined]
            )
            self.peak_host_memory_bytes = max(
                self.peak_host_memory_bytes,
                int(self.runtime.host_memory_bytes()),  # type: ignore[attr-defined]
            )
            print(
                "progress=local_minilm "
                f"forward_pairs={self.forward_pair_count} "
                f"forward_calls={self.forward_call_count} "
                f"elapsed_seconds={self.elapsed_seconds:.3f}",
                file=sys.stderr,
                flush=True,
            )
        return scores

    def execution_receipt(self) -> dict[str, object]:
        torch_module = self.runtime.torch  # type: ignore[attr-defined]
        return {
            "device": "cuda",
            "execution_backend": "rocm",
            "device_name": self.device_name,
            "torch_version": str(getattr(torch_module, "__version__", "unknown")),
            "torch_hip_version": str(getattr(torch_module.version, "hip", "unknown")),
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
            "peak_host_memory_bytes": self.peak_host_memory_bytes,
            "forward_pair_count": self.forward_pair_count,
            "forward_call_count": self.forward_call_count,
            "inference_elapsed_seconds": self.elapsed_seconds,
        }


def _default_inference_runtime() -> object:
    from types import SimpleNamespace

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return SimpleNamespace(
        torch=torch,
        auto_tokenizer_cls=AutoTokenizer,
        auto_model_cls=AutoModelForSequenceClassification,
        clock=time.perf_counter,
        host_memory_bytes=_host_memory_bytes,
    )


def _load_rocm_predictor(
    materialization: object,
    *,
    runtime: object | None = None,
) -> _RocmPredictor:
    runtime = runtime or _default_inference_runtime()
    torch_module = runtime.torch  # type: ignore[attr-defined]
    cuda = getattr(torch_module, "cuda", None)
    hip = getattr(getattr(torch_module, "version", None), "hip", None)
    if (
        not hip
        or cuda is None
        or not cuda.is_available()
        or int(cuda.device_count()) < 1
    ):
        raise RuntimeError("ROCm MiniLM scoring requires an available torch cuda device")
    device_name = str(cuda.get_device_name(0))
    snapshot = Path(getattr(materialization, "snapshot"))
    tokenizer = runtime.auto_tokenizer_cls.from_pretrained(  # type: ignore[attr-defined]
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    model = runtime.auto_model_cls.from_pretrained(  # type: ignore[attr-defined]
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch_module.float32,
    )
    model = model.float()  # type: ignore[attr-defined]
    model = model.eval()  # type: ignore[attr-defined]
    model = model.to("cuda")  # type: ignore[attr-defined]
    return _RocmPredictor(
        runtime=runtime,
        tokenizer=tokenizer,
        model=model,
        device_name=device_name,
    )


def _validate_model_binding(
    receipt: Mapping[str, object],
) -> object:
    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("score preflight bindings are missing")
    raw = bindings.get("model_materialization")
    if not isinstance(raw, Mapping):
        raise ValueError("score preflight model materialization binding is missing")
    receipt_path = raw.get("receipt_path")
    if not isinstance(receipt_path, str) or not receipt_path:
        raise ValueError("score preflight model materialization path is invalid")
    verified = load_verified_materialization(Path(receipt_path))
    payload = getattr(verified, "payload", None)
    if not isinstance(payload, Mapping):
        raise ValueError("verified model materialization payload is invalid")
    if (
        getattr(verified, "sha256", None) != raw.get("receipt_sha256")
        or payload.get("model_id") != MODEL_ID
        or payload.get("revision") != MODEL_REVISION
        or payload.get("snapshot_sha256") != raw.get("snapshot_sha256")
    ):
        raise ValueError("model materialization differs from score preflight")
    try:
        current = Path(getattr(verified, "receipt_path")).read_bytes()
    except (OSError, TypeError) as exc:
        raise ValueError("verified model receipt changed during scoring") from exc
    if current != getattr(verified, "source", None):
        raise ValueError("verified model receipt changed during scoring")
    return verified


def _load_bound_score_cache(
    receipt: Mapping[str, object], cache_root: Path
) -> GlobalScoreCache:
    bindings = receipt.get("bindings")
    assert isinstance(bindings, Mapping)
    raw = bindings.get("score_cache")
    if not isinstance(raw, Mapping):
        raise ValueError("score preflight cache binding is missing")
    context = score_cache_context()
    if raw.get("context") != context.artifact_metadata:
        raise ValueError("score preflight cache identity differs from pinned context")
    cache = GlobalScoreCache(Path(cache_root), context)
    expected_root = raw.get("root")
    expected_path = raw.get("path")
    if (
        not isinstance(expected_root, str)
        or Path(expected_root).resolve() != Path(cache_root).resolve()
        or not isinstance(expected_path, str)
        or Path(expected_path).resolve() != cache.path.resolve()
    ):
        raise ValueError("score cache path differs from preflight binding")
    return cache


def _verify_base_receipt(
    output_dir: Path,
    receipt: Mapping[str, object],
    queues: Mapping[tuple[str, str], Sequence[Mapping[str, object]]],
    *,
    preflight: Mapping[str, object],
    source: Mapping[str, object],
    preflight_sha256: str,
    windows_sha256: str,
) -> dict[str, object]:
    bindings = preflight.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("score preflight bindings are missing")
    materialization = bindings.get("model_materialization")
    cache_binding = bindings.get("score_cache")
    candidate_binding = source.get("candidates")
    if (
        not isinstance(materialization, Mapping)
        or not isinstance(cache_binding, Mapping)
        or not isinstance(candidate_binding, Mapping)
        or receipt.get("model_materialization_receipt_sha256")
        != materialization.get("receipt_sha256")
        or receipt.get("model_snapshot_sha256")
        != materialization.get("snapshot_sha256")
        or receipt.get("preflight_candidates_sha256")
        != candidate_binding.get("sha256")
        or receipt.get("score_cache_path") != cache_binding.get("path")
        or receipt.get("preflight_dir") != source.get("preflight_dir")
    ):
        raise ValueError("base score materialization or source binding differs")
    if (
        receipt.get("schema_version") != BASE_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("preflight_sha256") != preflight_sha256
        or receipt.get("preflight_windows_sha256") != windows_sha256
        or receipt.get("model") != MODEL_ID
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("model_loading")
        != {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "inference_dtype": "float32",
            "eval_mode": True,
        }
        or receipt.get("qrels_opened") is not False
        or receipt.get("network_call_count") != 0
        or receipt.get("retrieval_call_count") != 0
        or receipt.get("hosted_inference_call_count") != 0
        or receipt.get("paid_call_count") != 0
        or receipt.get("external_cost_usd") != 0.0
    ):
        raise ValueError("base score receipt binding or safety counters differ")
    for field in ("restored_cache_pair_count", "resumed_shard_count"):
        if field not in receipt:
            continue
        legacy_value = _require_nonnegative_int(
            receipt.get(field), f"legacy {field}"
        )
        if legacy_value != 0:
            raise ValueError(
                "base score receipt contains unauthenticated resume history"
            )
    receipt_forward_pairs = _require_nonnegative_int(
        receipt.get("unique_forward_pair_count"),
        "base receipt forward pair count",
    )
    receipt_forward_calls = _require_nonnegative_int(
        receipt.get("forward_call_count"),
        "base receipt forward call count",
    )
    _require_possible_forward_bookkeeping(
        receipt_forward_pairs,
        receipt_forward_calls,
        "base receipt",
    )
    raw_shards = receipt.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != len(queues):
        raise ValueError("base score receipt shard coverage differs")
    by_queue: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw in raw_shards:
        if not isinstance(raw, Mapping):
            raise ValueError("base score receipt shard entry is invalid")
        key = (str(raw.get("topic_id")), str(raw.get("obligation_id")))
        if key in by_queue:
            raise ValueError("base score receipt has duplicate shard coverage")
        by_queue[key] = raw
    if set(by_queue) != set(queues):
        raise ValueError("base score receipt obligation coverage differs")
    expected_artifacts: set[str] = set()
    expected_paths: dict[tuple[str, str], tuple[Path, Path]] = {}
    for key, raw in by_queue.items():
        path_value = raw.get("path")
        name = _shard_filename(*key)
        shard_path, sidecar_path = _confined_shard_paths(output_dir, name)
        if path_value != f"shards/{name}":
            raise ValueError("base score receipt shard path is invalid")
        expected_paths[key] = (shard_path, sidecar_path)
        expected_artifacts.add(shard_path.name)
        expected_artifacts.add(sidecar_path.name)
    try:
        observed_artifacts = {
            path.name for path in (output_dir / "shards").iterdir()
        }
    except OSError as exc:
        raise ValueError("base score shard directory is unreadable") from exc
    if observed_artifacts != expected_artifacts:
        raise ValueError("base score output contains an unexpected shard artifact")
    completed = 0
    total_forward_pairs = 0
    total_forward_calls = 0
    total_elapsed = 0.0
    for key, planned in queues.items():
        raw = by_queue[key]
        shard_path, sidecar = expected_paths[key]
        loaded = _load_complete_shard(
            shard_path,
            sidecar,
            planned,
            topic_id=key[0],
            obligation_id=key[1],
            preflight_sha256=preflight_sha256,
            windows_sha256=windows_sha256,
        )
        if loaded is None or dict(loaded) != dict(raw):
            raise ValueError("base score receipt differs from shard receipt")
        planned_pairs = {
            (str(row["query"]), str(row["window_text"])) for row in planned
        }
        unique_pairs = _require_nonnegative_int(
            loaded.get("unique_pair_count"), "shard unique pair count"
        )
        cache_hits = _require_nonnegative_int(
            loaded.get("cache_hit_pair_count"), "shard cache hit count"
        )
        cache_misses = _require_nonnegative_int(
            loaded.get("cache_miss_pair_count"), "shard cache miss count"
        )
        forward_pairs = _require_nonnegative_int(
            loaded.get("forward_pair_count"), "shard forward pair count"
        )
        forward_calls = _require_nonnegative_int(
            loaded.get("forward_call_count"), "shard forward call count"
        )
        _require_possible_forward_bookkeeping(
            forward_pairs,
            forward_calls,
            "shard",
        )
        shard_execution = loaded.get("execution")
        legacy_execution = receipt.get("model_constructed") is None
        if forward_pairs > 0 and not isinstance(shard_execution, Mapping):
            if not legacy_execution:
                raise ValueError(
                    "forward-bearing shard execution provenance is missing"
                )
        elif isinstance(shard_execution, Mapping):
            if (
                shard_execution.get("device") != "cuda"
                or shard_execution.get("execution_backend") != "rocm"
                or any(
                    not isinstance(shard_execution.get(name), str)
                    or not str(shard_execution[name])
                    for name in (
                        "device_name",
                        "torch_version",
                        "torch_hip_version",
                    )
                )
                or any(
                    isinstance(shard_execution.get(name), bool)
                    or not isinstance(shard_execution.get(name), int)
                    or int(shard_execution[name]) < 0
                    for name in (
                        "peak_device_memory_bytes",
                        "peak_host_memory_bytes",
                    )
                )
            ):
                raise ValueError("shard execution provenance is invalid")
        elif shard_execution is not None:
            raise ValueError("cache-only shard execution provenance is invalid")
        elapsed = loaded.get("elapsed_seconds")
        if (
            unique_pairs != len(planned_pairs)
            or cache_hits + cache_misses != unique_pairs
            or forward_pairs != cache_misses
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            raise ValueError("completed shard telemetry count differs")
        total_forward_pairs += forward_pairs
        total_forward_calls += forward_calls
        total_elapsed += float(elapsed)
        completed += len(planned)
    all_rows = [row for planned in queues.values() for row in planned]
    all_pairs = {
        (str(row["query"]), str(row["window_text"])) for row in all_rows
    }
    preflight_pair_hits: dict[tuple[str, str], bool] = {}
    for row in all_rows:
        preflight_pair_hits.setdefault(
            (str(row["query"]), str(row["window_text"])),
            bool(row["cache_hit"]),
        )
    initial_hits = _require_nonnegative_int(
        receipt.get("initial_runtime_cache_hit_pair_count"),
        "base receipt initial cache hit count",
    )
    initial_misses = _require_nonnegative_int(
        receipt.get("initial_runtime_cache_miss_pair_count"),
        "base receipt initial cache miss count",
    )
    summary = preflight.get("summary")
    if (
        not isinstance(summary, Mapping)
        or receipt.get("planned_document_count") != summary.get("document_count")
        or receipt.get("planned_candidate_count") != summary.get("candidate_count")
    ):
        raise ValueError("base score planned population differs from preflight")
    if (
        receipt.get("planned_window_count") != completed
        or receipt.get("completed_window_count") != completed
        or receipt.get("unique_pair_count") != len(all_pairs)
        or receipt.get("preflight_cache_hit_pair_count")
        != sum(preflight_pair_hits.values())
        or receipt.get("preflight_cache_miss_pair_count")
        != len(preflight_pair_hits) - sum(preflight_pair_hits.values())
        or initial_hits + initial_misses != len(all_pairs)
        or receipt.get("final_cache_hit_pair_count") != len(all_pairs)
        or receipt.get("final_cache_miss_pair_count") != 0
        or receipt_forward_pairs != total_forward_pairs
        or receipt_forward_calls != total_forward_calls
        or receipt.get("elapsed_seconds") != total_elapsed
        or receipt.get("shard_count") != len(queues)
        or total_forward_pairs > len(preflight_pair_hits) - sum(preflight_pair_hits.values())
    ):
        raise ValueError("base score receipt count or telemetry differs")
    expected_forward_rate = (
        total_forward_pairs / total_elapsed if total_elapsed > 0 else 0.0
    )
    expected_window_rate = completed / total_elapsed if total_elapsed > 0 else 0.0
    orchestration_elapsed = receipt.get("orchestration_elapsed_seconds")
    if (
        not isinstance(receipt.get("forward_pairs_per_second"), (int, float))
        or not math.isclose(
            float(receipt["forward_pairs_per_second"]),
            expected_forward_rate,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not isinstance(receipt.get("window_rows_per_second"), (int, float))
        or not math.isclose(
            float(receipt["window_rows_per_second"]),
            expected_window_rate,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or isinstance(orchestration_elapsed, bool)
        or not isinstance(orchestration_elapsed, (int, float))
        or not math.isfinite(float(orchestration_elapsed))
        or float(orchestration_elapsed) < 0
    ):
        raise ValueError("base score receipt throughput or duration differs")
    execution = _aggregate_shard_execution(by_queue)
    execution_fields = (
        "device",
        "execution_backend",
        "device_name",
        "torch_version",
        "torch_hip_version",
        "peak_device_memory_bytes",
        "peak_host_memory_bytes",
    )
    model_constructed = receipt.get("model_constructed")
    if execution is not None:
        if (
            model_constructed is not True
            or any(receipt.get(name) != execution.get(name) for name in execution_fields)
        ):
            raise ValueError("base score execution telemetry differs from its shards")
    elif total_forward_pairs == 0:
        cache_only = {
            "device": "cache",
            "execution_backend": "global_score_cache",
            "device_name": "not_applicable_cache_only",
            "torch_version": "not_loaded_cache_only",
            "torch_hip_version": "not_loaded_cache_only",
            "peak_device_memory_bytes": 0,
            "peak_host_memory_bytes": 0,
        }
        if model_constructed is not False or any(
            receipt.get(name) != value for name, value in cache_only.items()
        ):
            raise ValueError("base score cache-only execution state differs")
    elif model_constructed is not None or (
        receipt.get("device") != "cuda"
        or receipt.get("execution_backend") != "rocm"
        or any(
            not isinstance(receipt.get(name), str) or not str(receipt[name])
            for name in (
                "device_name",
                "torch_version",
                "torch_hip_version",
            )
        )
        or any(
            isinstance(receipt.get(name), bool)
            or not isinstance(receipt.get(name), int)
            or int(receipt[name]) < 0
            for name in (
                "peak_device_memory_bytes",
                "peak_host_memory_bytes",
            )
        )
    ):
        raise ValueError("base score execution telemetry is invalid")
    verified = dict(receipt)
    verified.pop("restored_cache_pair_count", None)
    verified.pop("resumed_shard_count", None)
    return verified


def run_local_scoring(
    preflight_dir: Path,
    output_dir: Path,
    cache_root: Path,
) -> dict[str, object]:
    """Score frozen queues locally, resuming only authenticated complete shards."""

    started = time.perf_counter()
    preflight, windows, source = _load_scoring_preflight(Path(preflight_dir))
    preflight_sha256 = str(source["preflight_sha256"])
    window_binding = source["windows"]
    assert isinstance(window_binding, Mapping)
    windows_sha256 = str(window_binding["sha256"])
    queues: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in windows:
        queues[(str(row["topic_id"]), str(row["variant"]))].append(row)

    planned_paths: dict[tuple[str, str], Path] = {}
    planned_sidecars: dict[tuple[str, str], Path] = {}
    seen_names: set[str] = set()
    destination = Path(output_dir).resolve(strict=False)
    for key in sorted(queues):
        name = _shard_filename(*key)
        if name in seen_names:
            raise ValueError("score shard names collide")
        seen_names.add(name)
        shard_path, sidecar_path = _confined_shard_paths(destination, name)
        planned_paths[key] = shard_path
        planned_sidecars[key] = sidecar_path

    if (destination / "receipt.json").exists():
        complete = _read_json(destination / "receipt.json", "base score receipt")
        return _verify_base_receipt(
            destination,
            complete,
            queues,
            preflight=preflight,
            source=source,
            preflight_sha256=preflight_sha256,
            windows_sha256=windows_sha256,
        )

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "shards").mkdir(exist_ok=True)
    expected_names = {
        path.name for path in planned_paths.values()
    } | {
        path.with_suffix(".receipt.json").name for path in planned_paths.values()
    }
    observed_names = {path.name for path in (destination / "shards").iterdir()}
    if not observed_names <= expected_names:
        raise ValueError("score output contains an unexpected shard artifact")

    shard_receipts: dict[tuple[str, str], dict[str, object]] = {}
    remaining: list[tuple[str, str]] = []
    for key, shard_path in planned_paths.items():
        loaded = _load_complete_shard(
            shard_path,
            planned_sidecars[key],
            queues[key],
            topic_id=key[0],
            obligation_id=key[1],
            preflight_sha256=preflight_sha256,
            windows_sha256=windows_sha256,
        )
        if loaded is None:
            remaining.append(key)
        else:
            shard_receipts[key] = loaded

    materialization = _validate_model_binding(preflight)
    cache = _load_bound_score_cache(preflight, Path(cache_root))
    for row in windows:
        expected_key = cache.cache_key(
            query_text=str(row["query"]),
            text=str(row["window_text"]),
        )
        if row.get("cache_key") != expected_key:
            raise ValueError("frozen cache key differs from the pinned cache context")
    all_pairs = {
        (str(row["query"]), str(row["window_text"])) for row in windows
    }
    initial_hits = sum(
        cache.get(query_text=query, text=text) is not None
        for query, text in all_pairs
    )
    for key in shard_receipts:
        restored: dict[tuple[str, str], float] = {}
        for row in _read_jsonl(planned_paths[key], "resumed score shard"):
            pair = (str(row["query"]), str(row["window_text"]))
            score = float(row["score"])
            if pair in restored and restored[pair] != score:
                raise ValueError("resumed shard has conflicting scores for one pair")
            restored[pair] = score
        cache.add_many(
            (query, text, score)
            for (query, text), score in restored.items()
        )
    predictor: object | None = None
    for key in remaining:
        planned = queues[key]
        unique_pairs = {
            (str(row["query"]), str(row["window_text"])) for row in planned
        }
        shard_hits = sum(
            cache.get(query_text=query, text=text) is not None
            for query, text in unique_pairs
        )
        shard_misses = len(unique_pairs) - shard_hits
        if shard_misses and predictor is None:
            predictor = _load_rocm_predictor(materialization)
        before_calls = int(getattr(predictor, "forward_call_count", 0))
        before_pairs = int(getattr(predictor, "forward_pair_count", 0))
        shard_started = time.perf_counter()
        scored = score_window_rows(
            planned,
            predict=predictor,  # type: ignore[arg-type]
            cache_get=lambda query, text: cache.get(query_text=query, text=text),
            cache_add=cache.add_many,
        )
        execution = _predictor_execution(predictor)
        score_rows = _planned_score_rows(
            scored,
            preflight_sha256=preflight_sha256,
            windows_sha256=windows_sha256,
            execution=execution,
        )
        score_bytes = _jsonl_bytes(score_rows)
        shard_path = planned_paths[key]
        _exclusive_bytes(shard_path, score_bytes)
        observed_calls = int(getattr(predictor, "forward_call_count", 0)) - before_calls
        observed_pairs = int(getattr(predictor, "forward_pair_count", 0)) - before_pairs
        shard_receipt: dict[str, object] = {
            "schema_version": BASE_SHARD_SCHEMA_VERSION,
            "status": "complete",
            "topic_id": key[0],
            "obligation_id": key[1],
            "path": f"shards/{shard_path.name}",
            "rows": len(score_rows),
            "bytes": len(score_bytes),
            "sha256": _sha256_bytes(score_bytes),
            "unique_pair_count": len(unique_pairs),
            "cache_hit_pair_count": shard_hits,
            "cache_miss_pair_count": shard_misses,
            "forward_pair_count": observed_pairs if predictor is not None else 0,
            "forward_call_count": observed_calls if predictor is not None else 0,
            "execution": dict(execution) if shard_misses else None,
            "elapsed_seconds": time.perf_counter() - shard_started,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "preflight_sha256": preflight_sha256,
            "preflight_windows_sha256": windows_sha256,
            "qrels_opened": False,
            "network_call_count": 0,
            "paid_call_count": 0,
        }
        if shard_misses and (observed_pairs != shard_misses or observed_calls <= 0):
            raise ValueError("ROCm predictor telemetry differs from scored misses")
        _exclusive_json(planned_sidecars[key], shard_receipt)
        shard_receipts[key] = shard_receipt

    if set(shard_receipts) != set(queues):
        raise ValueError("base score shards do not cover every obligation")
    execution = _aggregate_shard_execution(shard_receipts)
    if execution is None:
        execution = _predictor_execution(predictor)
    final_hits = sum(cache.get(query_text=query, text=text) is not None for query, text in all_pairs)
    total_forward_pairs = sum(int(row["forward_pair_count"]) for row in shard_receipts.values())
    total_forward_calls = sum(int(row["forward_call_count"]) for row in shard_receipts.values())
    scoring_elapsed = sum(float(row["elapsed_seconds"]) for row in shard_receipts.values())
    receipt: dict[str, object] = {
        "schema_version": BASE_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_materialization_receipt_sha256": getattr(materialization, "sha256"),
        "model_snapshot_sha256": getattr(materialization, "payload")["snapshot_sha256"],
        "model_loading": {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "inference_dtype": "float32",
            "eval_mode": True,
        },
        "model_constructed": execution.get("execution_backend") == "rocm",
        "device": execution.get("device", "cuda"),
        "execution_backend": execution.get("execution_backend", "rocm"),
        "device_name": execution.get("device_name"),
        "torch_version": execution.get("torch_version"),
        "torch_hip_version": execution.get("torch_hip_version"),
        "peak_device_memory_bytes": int(execution.get("peak_device_memory_bytes", 0)),
        "peak_host_memory_bytes": int(execution.get("peak_host_memory_bytes", 0)),
        "preflight_dir": str(Path(preflight_dir).resolve()),
        "preflight_sha256": preflight_sha256,
        "preflight_candidates_sha256": source["candidates"]["sha256"],  # type: ignore[index]
        "preflight_windows_sha256": windows_sha256,
        "planned_document_count": preflight["summary"]["document_count"],  # type: ignore[index]
        "planned_candidate_count": preflight["summary"]["candidate_count"],  # type: ignore[index]
        "planned_window_count": len(windows),
        "completed_window_count": sum(int(row["rows"]) for row in shard_receipts.values()),
        "unique_pair_count": len(all_pairs),
        "preflight_cache_hit_pair_count": preflight["summary"]["cache_hit_count"],  # type: ignore[index]
        "preflight_cache_miss_pair_count": preflight["summary"]["cache_miss_count"],  # type: ignore[index]
        "initial_runtime_cache_hit_pair_count": initial_hits,
        "initial_runtime_cache_miss_pair_count": len(all_pairs) - initial_hits,
        "final_cache_hit_pair_count": final_hits,
        "final_cache_miss_pair_count": len(all_pairs) - final_hits,
        "unique_forward_pair_count": total_forward_pairs,
        "forward_call_count": total_forward_calls,
        "elapsed_seconds": scoring_elapsed,
        "orchestration_elapsed_seconds": time.perf_counter() - started,
        "forward_pairs_per_second": (
            total_forward_pairs / scoring_elapsed if scoring_elapsed > 0 else 0.0
        ),
        "window_rows_per_second": (
            len(windows) / scoring_elapsed if scoring_elapsed > 0 else 0.0
        ),
        "score_cache_path": str(cache.path.resolve()),
        "shard_count": len(shard_receipts),
        "shards": [shard_receipts[key] for key in sorted(shard_receipts)],
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "external_cost_usd": 0.0,
    }
    if receipt["completed_window_count"] != len(windows) or final_hits != len(all_pairs):
        raise ValueError("base scoring did not cover every frozen window pair")
    _exclusive_json(destination / "receipt.json", receipt)
    return _verify_base_receipt(
        destination,
        receipt,
        queues,
        preflight=preflight,
        source=source,
        preflight_sha256=preflight_sha256,
        windows_sha256=windows_sha256,
    )


def verify_local_scoring(output_dir: Path) -> dict[str, object]:
    """Authenticate a completed base-scoring receipt and every frozen shard."""

    destination = Path(output_dir)
    receipt = _read_json(destination / "receipt.json", "base score receipt")
    preflight_value = receipt.get("preflight_dir")
    if not isinstance(preflight_value, str) or not preflight_value:
        raise ValueError("base score receipt preflight path is missing")
    preflight, windows, source = _load_scoring_preflight(Path(preflight_value))
    queues: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in windows:
        queues[(str(row["topic_id"]), str(row["variant"]))].append(row)
    window_binding = source["windows"]
    assert isinstance(window_binding, Mapping)
    return _verify_base_receipt(
        destination,
        receipt,
        queues,
        preflight=preflight,
        source=source,
        preflight_sha256=str(source["preflight_sha256"]),
        windows_sha256=str(window_binding["sha256"]),
    )


def _file_binding(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {
            "state": "absent",
            "path": str(resolved),
            "bytes": 0,
            "sha256": None,
        }
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise ValueError(f"bound input is not a readable regular file: {resolved}")
    return {
        "state": "present",
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def persist_score_preflight(
    output_dir: Path,
    *,
    candidates: Sequence[Mapping[str, object]],
    preflight: Mapping[str, object],
    bindings: Mapping[str, object],
) -> dict[str, object]:
    """Persist create-only candidates, windows, and their no-inference receipt."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"create-only preflight output exists: {destination}")
    raw_windows = preflight.get("windows")
    if not isinstance(raw_windows, list) or any(
        not isinstance(row, Mapping) for row in raw_windows
    ):
        raise ValueError("preflight windows must be an array of objects")
    summary = preflight.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("preflight summary must be an object")
    candidate_bytes = _jsonl_bytes(candidates)
    window_bytes = _jsonl_bytes(raw_windows)  # type: ignore[arg-type]
    unique_misses = int(summary["cache_miss_count"])
    projected = REFERENCE_FIXED_SECONDS + (
        unique_misses / REFERENCE_PAIRS_PER_SECOND
    )
    receipt: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "tokenizer_only_preflight_complete",
        "qrels_opened": False,
        "network": False,
        "external_cost_usd": 0.0,
        "inference_count": 0,
        "inference_authorized": False,
        "model_constructed": False,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "bindings": dict(bindings),
        "artifacts": {
            "candidates.jsonl": {
                "rows": len(candidates),
                "bytes": len(candidate_bytes),
                "sha256": _sha256_bytes(candidate_bytes),
            },
            "windows.jsonl": {
                "rows": len(raw_windows),
                "bytes": len(window_bytes),
                "sha256": _sha256_bytes(window_bytes),
            },
        },
        "window_policy": {
            "version": WINDOW_POLICY_VERSION,
            "pair_max_tokens": PAIR_MAX_TOKENS,
            "query_max_tokens": QUERY_MAX_TOKENS,
            "minimum_passage_tokens": MIN_PASSAGE_TOKENS,
            "passage_overlap_tokens": PASSAGE_OVERLAP_TOKENS,
            "maximum_windows_per_document": MAX_WINDOWS_PER_DOCUMENT,
        },
        "summary": dict(summary),
        "coverage_matrix": {
            "topics": list(preflight.get("topics", [])),
            "obligations": list(preflight.get("obligations", [])),
            "documents": list(preflight.get("documents", [])),
        },
        "projection_basis": {
            "pairs_per_second": REFERENCE_PAIRS_PER_SECOND,
            "fixed_seconds": REFERENCE_FIXED_SECONDS,
            "pair_count": unique_misses,
            "device_class": "local_rocm",
        },
        "projected_rocm_runtime_seconds": projected,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    for name, content in (
        ("candidates.jsonl", candidate_bytes),
        ("windows.jsonl", window_bytes),
        ("preflight.json", _pretty_json_bytes(receipt)),
    ):
        with (destination / name).open("xb") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
    return receipt


def run_score_preflight(
    *,
    contract_dir: Path,
    output_dir: Path,
    model_receipt_path: Path | None = None,
    score_cache_root: Path | None = None,
) -> dict[str, object]:
    """Authenticate local inputs and materialize the score-coverage preflight."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"create-only preflight output exists: {destination}")
    contract = load_score_contract(Path(contract_dir))
    candidates = build_score_candidates(contract)

    repo_root = find_repo_root()
    materialization_path = Path(
        model_receipt_path or repo_root / DEFAULT_MODEL_RECEIPT
    ).resolve()
    materialization = load_verified_materialization(materialization_path)
    if (
        materialization.payload.get("model_id") != MODEL_ID
        or materialization.payload.get("revision") != MODEL_REVISION
    ):
        raise ValueError("materialized model identity differs from pinned MiniLM")
    tokenizer = load_verified_tokenizer(materialization_path)

    context = score_cache_context()
    cache_root = Path(
        score_cache_root or repo_cache_root(repo_root) / "reranker"
    ).resolve()
    expected_cache_path = cache_root.joinpath(*context.path_parts)
    cache_binding = _file_binding(expected_cache_path)
    cache = GlobalScoreCache(cache_root, context)
    if cache.path.resolve() != expected_cache_path.resolve():
        raise ValueError("score cache path differs from the pinned cache context")
    if _file_binding(expected_cache_path) != cache_binding:
        raise ValueError("score cache changed while it was being loaded")
    plan = build_score_preflight(
        candidates,
        tokenizer,
        cache_lookup=lambda query, text: cache.get(query_text=query, text=text),
    )
    if materialization.receipt_path.read_bytes() != materialization.source:
        raise ValueError("model materialization receipt changed during preflight")
    if _file_binding(expected_cache_path) != cache_binding:
        raise ValueError("score cache changed during preflight")

    contract_summary_path = Path(contract_dir) / "summary.json"
    contract_summary = _read_json(contract_summary_path, "contract summary")
    source_files = {
        "adaptive_evidence_score.py": Path(__file__),
        "adaptive_evidence_contract.py": Path(__file__).with_name(
            "adaptive_evidence_contract.py"
        ),
        "facet_local_minilm_preflight.py": Path(__file__).with_name(
            "facet_local_minilm_preflight.py"
        ),
        "rerank_score_cache.py": Path(__file__).with_name("rerank_score_cache.py"),
    }
    bindings: dict[str, object] = {
        "contract": {
            "path": str(Path(contract_dir).resolve()),
            "summary_sha256": _sha256_file(contract_summary_path),
            "artifact_sha256": contract_summary["artifact_sha256"],
        },
        "model_materialization": {
            "receipt_path": str(materialization.receipt_path),
            "receipt_sha256": materialization.sha256,
            "snapshot_sha256": materialization.payload["snapshot_sha256"],
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_fast": True,
        },
        "score_cache": {
            "root": str(cache_root),
            "path": str(cache.path.resolve()),
            "context": context.artifact_metadata,
            "binding": cache_binding,
        },
        "code_sha256": {
            name: _sha256_file(path) for name, path in source_files.items()
        },
    }
    return persist_score_preflight(
        destination,
        candidates=candidates,
        preflight=plan,
        bindings=bindings,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="audit local MiniLM score coverage without inference"
    )
    preflight.add_argument("--contract", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser(
        "score", help="score frozen windows with the authenticated local ROCm model"
    )
    score.add_argument("--preflight", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--cache-root", type=Path)
    verify = subparsers.add_parser(
        "verify", help="authenticate every completed base-score shard"
    )
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        receipt = run_score_preflight(
            contract_dir=args.contract,
            output_dir=args.output,
        )
        summary = receipt["summary"]
        assert isinstance(summary, Mapping)
        print(
            "status=complete "
            f"documents={summary['document_count']} "
            f"candidates={summary['candidate_count']} "
            f"windows={summary['window_count']} "
            f"pairs={summary['unique_pair_count']} "
            f"cache_hits={summary['cache_hit_count']} "
            f"cache_misses={summary['cache_miss_count']} "
            f"projected_rocm_seconds={receipt['projected_rocm_runtime_seconds']:.3f} "
            "network=false qrels_opened=false external_cost=$0"
        )
    elif args.command == "score":
        repo_root = find_repo_root()
        cache_root = Path(
            args.cache_root or repo_cache_root(repo_root) / "reranker"
        )
        receipt = run_local_scoring(args.preflight, args.output, cache_root)
        print(
            "status=complete "
            f"windows={receipt['completed_window_count']} "
            f"pairs={receipt['unique_pair_count']} "
            f"forwards={receipt['unique_forward_pair_count']} "
            f"shards={receipt['shard_count']} "
            f"elapsed_seconds={receipt['elapsed_seconds']:.3f} "
            "network=false qrels_opened=false external_cost=$0"
        )
    else:
        receipt = verify_local_scoring(args.output)
        print(
            "status=verified "
            f"windows={receipt['completed_window_count']} "
            f"pairs={receipt['unique_pair_count']} "
            f"shards={receipt['shard_count']} "
            "network=false qrels_opened=false external_cost=$0"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
