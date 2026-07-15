"""Build and authenticate complete qrels-blind baseline candidate rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .adaptive_evidence_discovery import verify_discovery_terminal
from .adaptive_evidence_score import load_score_contract, verify_local_scoring


SCHEMA_VERSION = "adaptive-evidence-baseline-rankings-v1"
ROW_SCHEMA_VERSION = "adaptive-evidence-baseline-ranking-row-v1"
EXPECTED_DOCUMENT_COUNT = 8_114
EXPECTED_BROAD_COUNT = 4
EXPECTED_O0_COUNT = 24
EXPECTED_WINDOW_COUNT = 98_053
EXPECTED_PAIR_COUNT = 96_911
EXPECTED_SHARD_COUNT = 28

_AVAILABLE_ARMS = ("NARRATIVE", "FIXED-O0")
_UNAVAILABLE_ARMS = ("ADAPTIVE", "COMPOSITE")
_RANKING_NAMES = {
    "NARRATIVE": "narrative.jsonl",
    "FIXED-O0": "fixed_o0.jsonl",
}
_OUTPUT_NAMES = frozenset({*_RANKING_NAMES.values(), "availability.json", "receipt.json"})
_ZERO_COUNTERS = (
    "network_call_count",
    "retrieval_call_count",
    "hosted_inference_call_count",
    "paid_call_count",
    "model_load_count",
    "inference_count",
)
_REQUIRED_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "topic_id",
        "topic_rank",
        "document_id",
        "window_id",
        "window_text",
        "primary_obligation",
        "score_scope",
        "score",
        "selection_reason",
        "union_order",
        "passage_tokens",
        "scored_obligations",
        "document_sha256",
        "window_sha256",
        "query_sha256",
        "preflight_sha256",
        "preflight_windows_sha256",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_compact_bytes(dict(row)) for row in rows)


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
        if _compact_bytes(row).rstrip(b"\n") != line:
            raise ValueError(f"{label}:{line_number} is not canonical JSON")
        rows.append(row)
    return rows


def _reject_protected(topic_ids: Sequence[object]) -> None:
    for topic_id in map(str, topic_ids):
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _data_value(data: object, name: str) -> object:
    if isinstance(data, Mapping):
        if name not in data:
            raise ValueError(f"ranking data {name} is missing")
        return data[name]
    try:
        return getattr(data, name)
    except AttributeError as exc:
        raise ValueError(f"ranking data {name} is missing") from exc


def _records(data: object, name: str) -> list[dict[str, object]]:
    value = _data_value(data, name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"ranking data {name} must be an array")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"ranking data {name} must contain objects")
    return [dict(row) for row in value]  # type: ignore[arg-type]


def _topic_ids(data: object) -> list[str]:
    value = _data_value(data, "topic_ids")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("ranking data topic_ids must be an array")
    topic_ids = [str(topic_id) for topic_id in value]
    _reject_protected(topic_ids)
    if not topic_ids or len(topic_ids) != len(set(topic_ids)) or any(not value for value in topic_ids):
        raise ValueError("ranking topic IDs must be distinct nonempty text")
    return topic_ids


def _obligation_id(row: Mapping[str, object]) -> str:
    value = row.get("variant", row.get("obligation_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("score obligation identity must be nonempty text")
    return value


def best_window_per_document(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Reduce windows only within one topic/obligation/document score queue."""

    best: dict[tuple[str, str, str], tuple[tuple[float, int, str], dict[str, object]]] = {}
    for source in rows:
        row = dict(source)
        topic_id = row.get("topic_id")
        document_id = row.get("document_id")
        if not isinstance(topic_id, str) or not topic_id:
            raise ValueError("score topic_id must be nonempty text")
        _reject_protected([topic_id])
        obligation_id = _obligation_id(row)
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("score document_id must be nonempty text")
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("MiniLM score must be numeric")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("MiniLM score is nonfinite")
        start = row.get("document_start_token")
        window_id = row.get("window_id")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("window document_start_token must be nonnegative")
        if not isinstance(window_id, str) or not window_id:
            raise ValueError("window_id must be nonempty text")
        key = (topic_id, obligation_id, document_id)
        rank_key = (-value, start, window_id)
        if key not in best or rank_key < best[key][0]:
            best[key] = (rank_key, row)
    return [best[key][1] for key in sorted(best)]


def _normalized_data(data: object) -> dict[str, object]:
    topic_ids = _topic_ids(data)
    topic_set = set(topic_ids)
    obligations = _records(data, "obligations")
    documents = _records(data, "documents")
    score_rows = _records(data, "score_rows")

    obligation_by_id: dict[str, dict[str, object]] = {}
    obligations_by_topic: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in obligations:
        topic_id = row.get("topic_id")
        obligation_id = row.get("obligation_id")
        kind = row.get("kind")
        manifest_order = row.get("manifest_order")
        if (
            not isinstance(topic_id, str)
            or topic_id not in topic_set
            or not isinstance(obligation_id, str)
            or not obligation_id
            or obligation_id in obligation_by_id
            or kind not in {"broad", "o0"}
            or isinstance(manifest_order, bool)
            or not isinstance(manifest_order, int)
        ):
            raise ValueError("ranking obligation schema or identity is invalid")
        obligation_by_id[obligation_id] = row
        obligations_by_topic[topic_id].append(row)
    for topic_id in topic_ids:
        local = obligations_by_topic[topic_id]
        if sum(row["kind"] == "broad" for row in local) != 1:
            raise ValueError(f"topic {topic_id} must have exactly one broad obligation")
        o0_orders = [int(row["manifest_order"]) for row in local if row["kind"] == "o0"]
        if len(o0_orders) != len(set(o0_orders)):
            raise ValueError(f"topic {topic_id} O0 manifest order is not unique")

    document_by_key: dict[tuple[str, str], dict[str, object]] = {}
    documents_by_topic: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in documents:
        topic_id = row.get("topic_id")
        document_id = row.get("document_id")
        union_order = row.get("union_order")
        text = row.get("text")
        text_sha256 = row.get("text_sha256")
        if (
            not isinstance(topic_id, str)
            or topic_id not in topic_set
            or not isinstance(document_id, str)
            or not document_id
            or isinstance(union_order, bool)
            or not isinstance(union_order, int)
            or union_order < 1
            or not isinstance(text, str)
            or text_sha256 != _sha256_bytes(text.encode("utf-8"))
        ):
            raise ValueError("ranking document schema, order, or text hash is invalid")
        key = (topic_id, document_id)
        if key in document_by_key:
            raise ValueError("ranking documents contain a duplicate identity")
        document_by_key[key] = row
        documents_by_topic[topic_id].append(row)
    for topic_id in topic_ids:
        local_orders = [int(row["union_order"]) for row in documents_by_topic[topic_id]]
        if not local_orders or len(local_orders) != len(set(local_orders)):
            raise ValueError(f"topic {topic_id} union order is missing or duplicated")

    best_rows = best_window_per_document(score_rows)
    best_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in best_rows:
        topic_id = str(row["topic_id"])
        obligation_id = _obligation_id(row)
        document_id = str(row["document_id"])
        obligation = obligation_by_id.get(obligation_id)
        document = document_by_key.get((topic_id, document_id))
        start = row.get("document_start_token")
        end = row.get("document_end_token")
        window_text = row.get("window_text")
        if (
            obligation is None
            or obligation.get("topic_id") != topic_id
            or document is None
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end <= start
            or not isinstance(window_text, str)
            or row.get("document_sha256") != document.get("text_sha256")
            or row.get("window_sha256") != _sha256_bytes(window_text.encode("utf-8"))
            or any(
                not _is_sha256(row.get(name))
                for name in (
                    "query_sha256",
                    "preflight_sha256",
                    "preflight_windows_sha256",
                )
            )
        ):
            raise ValueError("best score window provenance or immutable hash is invalid")
        best_by_key[(topic_id, obligation_id, document_id)] = row

    queue_ids_by_topic: dict[str, list[str]] = {}
    queue_rows: dict[tuple[str, str], list[dict[str, object]]] = {}
    scored_by_document: dict[tuple[str, str], list[str]] = {}
    for topic_id in topic_ids:
        broad = next(
            row for row in obligations_by_topic[topic_id] if row["kind"] == "broad"
        )
        o0 = sorted(
            (row for row in obligations_by_topic[topic_id] if row["kind"] == "o0"),
            key=lambda row: (int(row["manifest_order"]), str(row["obligation_id"])),
        )
        queue_ids = [str(broad["obligation_id"]), *(str(row["obligation_id"]) for row in o0)]
        queue_ids_by_topic[topic_id] = queue_ids
        union_order = {
            str(row["document_id"]): int(row["union_order"])
            for row in documents_by_topic[topic_id]
        }
        for obligation_id in queue_ids:
            local = [
                row
                for (row_topic, row_obligation, _document_id), row in best_by_key.items()
                if row_topic == topic_id and row_obligation == obligation_id
            ]
            queue_rows[(topic_id, obligation_id)] = sorted(
                local,
                key=lambda row: (
                    -float(row["score"]),
                    union_order[str(row["document_id"])],
                    str(row["document_id"]),
                ),
            )
        broad_ids = {
            str(row["document_id"])
            for row in queue_rows[(topic_id, str(broad["obligation_id"]))]
        }
        expected_ids = {str(row["document_id"]) for row in documents_by_topic[topic_id]}
        if broad_ids != expected_ids:
            raise ValueError(f"topic {topic_id} broad score queue does not cover every document")
        for document_id in expected_ids:
            scored_by_document[(topic_id, document_id)] = [
                obligation_id
                for obligation_id in queue_ids
                if (topic_id, obligation_id, document_id) in best_by_key
            ]
    return {
        "topic_ids": topic_ids,
        "obligations": obligations,
        "documents": documents,
        "documents_by_topic": documents_by_topic,
        "queue_ids_by_topic": queue_ids_by_topic,
        "queue_rows": queue_rows,
        "scored_by_document": scored_by_document,
    }


def _output_row(
    *,
    source: Mapping[str, object],
    topic_rank: int,
    obligation_id: str,
    selection_reason: str,
    union_order: int,
    scored_obligations: Sequence[str],
) -> dict[str, object]:
    start = int(source["document_start_token"])
    end = int(source["document_end_token"])
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "topic_id": str(source["topic_id"]),
        "topic_rank": topic_rank,
        "document_id": str(source["document_id"]),
        "window_id": str(source["window_id"]),
        "window_text": str(source["window_text"]),
        "primary_obligation": obligation_id,
        "score_scope": obligation_id,
        "score": float(source["score"]),
        "selection_reason": selection_reason,
        "union_order": union_order,
        "passage_tokens": end - start,
        "scored_obligations": list(scored_obligations),
        "document_sha256": str(source["document_sha256"]),
        "window_sha256": str(source["window_sha256"]),
        "query_sha256": str(source["query_sha256"]),
        "preflight_sha256": str(source["preflight_sha256"]),
        "preflight_windows_sha256": str(source["preflight_windows_sha256"]),
    }


def build_narrative_continuation(data: object) -> list[dict[str, object]]:
    """Rank every topic document by its own broad-query best window."""

    normalized = _normalized_data(data)
    rows: list[dict[str, object]] = []
    documents_by_topic = normalized["documents_by_topic"]
    queue_ids_by_topic = normalized["queue_ids_by_topic"]
    queue_rows = normalized["queue_rows"]
    scored = normalized["scored_by_document"]
    assert isinstance(documents_by_topic, Mapping)
    assert isinstance(queue_ids_by_topic, Mapping)
    assert isinstance(queue_rows, Mapping)
    assert isinstance(scored, Mapping)
    for topic_id in normalized["topic_ids"]:  # type: ignore[union-attr]
        queue_ids = queue_ids_by_topic[topic_id]
        broad_id = queue_ids[0]
        union_order = {
            str(row["document_id"]): int(row["union_order"])
            for row in documents_by_topic[topic_id]
        }
        for topic_rank, source in enumerate(queue_rows[(topic_id, broad_id)], start=1):
            document_id = str(source["document_id"])
            rows.append(
                _output_row(
                    source=source,
                    topic_rank=topic_rank,
                    obligation_id=broad_id,
                    selection_reason="broad_score_descending",
                    union_order=union_order[document_id],
                    scored_obligations=scored[(topic_id, document_id)],
                )
            )
    return rows


def _next_unselected(
    queue: Sequence[Mapping[str, object]], selected: set[str]
) -> Mapping[str, object] | None:
    return next(
        (row for row in queue if str(row["document_id"]) not in selected),
        None,
    )


def build_fixed_o0_continuation(data: object) -> list[dict[str, object]]:
    """Build broad/O0 coverage, token-deficit rotation, then the broad tail."""

    normalized = _normalized_data(data)
    rows: list[dict[str, object]] = []
    documents_by_topic = normalized["documents_by_topic"]
    queue_ids_by_topic = normalized["queue_ids_by_topic"]
    queue_rows = normalized["queue_rows"]
    scored = normalized["scored_by_document"]
    assert isinstance(documents_by_topic, Mapping)
    assert isinstance(queue_ids_by_topic, Mapping)
    assert isinstance(queue_rows, Mapping)
    assert isinstance(scored, Mapping)
    for topic_id in normalized["topic_ids"]:  # type: ignore[union-attr]
        queue_ids = list(queue_ids_by_topic[topic_id])
        broad_id = queue_ids[0]
        facet_ids = queue_ids[1:]
        union_order = {
            str(row["document_id"]): int(row["union_order"])
            for row in documents_by_topic[topic_id]
        }
        selected: set[str] = set()
        token_totals = {obligation_id: 0 for obligation_id in queue_ids}
        topic_output: list[dict[str, object]] = []

        def select(source: Mapping[str, object], obligation_id: str, reason: str) -> None:
            document_id = str(source["document_id"])
            if document_id in selected:
                raise ValueError("fixed-O0 attempted to select a duplicate document")
            selected.add(document_id)
            token_totals[obligation_id] += int(source["document_end_token"]) - int(
                source["document_start_token"]
            )
            topic_output.append(
                _output_row(
                    source=source,
                    topic_rank=len(topic_output) + 1,
                    obligation_id=obligation_id,
                    selection_reason=reason,
                    union_order=union_order[document_id],
                    scored_obligations=scored[(topic_id, document_id)],
                )
            )

        for obligation_id in queue_ids:
            source = _next_unselected(queue_rows[(topic_id, obligation_id)], selected)
            if source is not None:
                select(source, obligation_id, "coverage_floor")

        priority = {obligation_id: index for index, obligation_id in enumerate(queue_ids)}
        while True:
            eligible_facets = [
                obligation_id
                for obligation_id in facet_ids
                if _next_unselected(queue_rows[(topic_id, obligation_id)], selected)
                is not None
            ]
            if not eligible_facets:
                break
            eligible = [*eligible_facets]
            if _next_unselected(queue_rows[(topic_id, broad_id)], selected) is not None:
                eligible.append(broad_id)
            obligation_id = min(
                eligible,
                key=lambda value: (token_totals[value], priority[value], value),
            )
            source = _next_unselected(queue_rows[(topic_id, obligation_id)], selected)
            assert source is not None
            select(source, obligation_id, "token_deficit_round_robin")

        for source in queue_rows[(topic_id, broad_id)]:
            if str(source["document_id"]) not in selected:
                select(source, broad_id, "broad_tail")
        if len(topic_output) != len(documents_by_topic[topic_id]):
            raise ValueError(f"topic {topic_id} fixed-O0 ranking is incomplete")
        rows.extend(topic_output)
    return rows


def _safe_receipt(receipt: Mapping[str, object], label: str) -> None:
    if (
        receipt.get("qrels_opened") is not False
        or any(receipt.get(name) != 0 for name in _ZERO_COUNTERS[:4])
        or receipt.get("external_cost_usd") != 0.0
    ):
        raise ValueError(f"{label} violates the frozen safety receipt")


def load_authenticated_ranking_data(
    *,
    contract_dir: Path,
    scores_dir: Path,
    discovery_dir: Path,
) -> dict[str, object]:
    """Authenticate Tasks 1, 3, and terminal Task 4 before loading score rows."""

    contract_root = Path(contract_dir)
    scores_root = Path(scores_dir)
    discovery_root = Path(discovery_dir)
    # load_score_contract rejects manifest topic metadata before documents.jsonl.
    contract = load_score_contract(contract_root)
    obligations = list(contract["obligations"])  # type: ignore[index]
    documents = list(contract["documents"])  # type: ignore[index]
    _reject_protected(
        [row.get("topic_id") for row in obligations]
        + [row.get("topic_id") for row in documents]
    )
    score_receipt = verify_local_scoring(scores_root)
    discovery_receipt = verify_discovery_terminal(discovery_root)
    _safe_receipt(score_receipt, "base scores")
    _safe_receipt(discovery_receipt, "terminal discovery")
    if (
        len(documents) != EXPECTED_DOCUMENT_COUNT
        or sum(row.get("kind") == "broad" for row in obligations) != EXPECTED_BROAD_COUNT
        or sum(row.get("kind") == "o0" for row in obligations) != EXPECTED_O0_COUNT
        or score_receipt.get("completed_window_count") != EXPECTED_WINDOW_COUNT
        or score_receipt.get("unique_pair_count") != EXPECTED_PAIR_COUNT
        or score_receipt.get("shard_count") != EXPECTED_SHARD_COUNT
        or discovery_receipt.get("status") != "discovery_unavailable"
    ):
        raise ValueError("authenticated ranking source counts differ from the canonical inputs")
    topic_ids = [str(topic_id) for topic_id in PILOT_TOPIC_IDS]
    score_rows: list[dict[str, object]] = []
    raw_shards = score_receipt.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("base score shard inventory is missing")
    for shard in raw_shards:
        if not isinstance(shard, Mapping):
            raise ValueError("base score shard inventory is invalid")
        relative = shard.get("path")
        if (
            not isinstance(relative, str)
            or not relative.startswith("shards/")
            or Path(relative).parts[:1] != ("shards",)
        ):
            raise ValueError("base score shard path is invalid")
        score_rows.extend(_read_jsonl(scores_root / relative, "authenticated score shard"))
    if len(score_rows) != EXPECTED_WINDOW_COUNT:
        raise ValueError("authenticated score shard population differs")
    _reject_protected([row.get("topic_id") for row in score_rows])
    return {
        "topic_ids": topic_ids,
        "obligations": obligations,
        "documents": documents,
        "score_rows": score_rows,
        "source_bindings": {
            "mode": "authenticated_paths",
            "contract_dir": str(contract_root.resolve()),
            "contract_summary_sha256": _sha256_file(contract_root / "summary.json"),
            "base_scores_dir": str(scores_root.resolve()),
            "base_score_receipt_sha256": _sha256_file(scores_root / "receipt.json"),
            "discovery_dir": str(discovery_root.resolve()),
            "terminal_discovery_receipt_sha256": _sha256_file(
                discovery_root / "receipt.json"
            ),
        },
        "discovery_receipt": discovery_receipt,
    }


def _source_bindings(data: object) -> dict[str, object]:
    try:
        raw = _data_value(data, "source_bindings")
    except ValueError:
        raw = {
            "mode": "provided_data",
            "contract_summary_sha256": "0" * 64,
            "base_score_receipt_sha256": "0" * 64,
            "terminal_discovery_receipt_sha256": "0" * 64,
        }
    if not isinstance(raw, Mapping):
        raise ValueError("ranking source bindings must be an object")
    bindings = dict(raw)
    if bindings.get("mode") not in {"provided_data", "authenticated_paths"} or any(
        not _is_sha256(bindings.get(name))
        for name in (
            "contract_summary_sha256",
            "base_score_receipt_sha256",
            "terminal_discovery_receipt_sha256",
        )
    ):
        raise ValueError("ranking source receipt hashes are invalid")
    return bindings


def _artifact_record(content: bytes, rows: int, path: str) -> dict[str, object]:
    return {
        "path": path,
        "rows": rows,
        "bytes": len(content),
        "sha256": _sha256_bytes(content),
    }


def _unavailable_arm(terminal_sha256: str) -> dict[str, object]:
    return {
        "status": "unavailable",
        "reason": "terminal_discovery_unavailable",
        "terminal_discovery_receipt_sha256": terminal_sha256,
        "o1_obligation_count": 0,
        "n1_nugget_count": 0,
        "proposed_o1_count": 0,
        "validated_o1_count": 0,
        "accepted_o1_count": 0,
        "proposed_n1_count": 0,
        "validated_n1_count": 0,
        "accepted_n1_count": 0,
    }


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())


def build_rankings(data: object, *, output_dir: Path) -> dict[str, object]:
    """Create the two complete baselines and explicit arm availability once."""

    destination = Path(output_dir)
    if _path_present(destination):
        raise FileExistsError(f"create-only ranking output exists: {destination}")
    narrative = build_narrative_continuation(data)
    fixed = build_fixed_o0_continuation(data)
    topic_ids = _topic_ids(data)
    bindings = _source_bindings(data)
    terminal_sha256 = str(bindings["terminal_discovery_receipt_sha256"])
    narrative_bytes = _jsonl_bytes(narrative)
    fixed_bytes = _jsonl_bytes(fixed)
    ranking_records = {
        "NARRATIVE": _artifact_record(
            narrative_bytes, len(narrative), _RANKING_NAMES["NARRATIVE"]
        ),
        "FIXED-O0": _artifact_record(
            fixed_bytes, len(fixed), _RANKING_NAMES["FIXED-O0"]
        ),
    }
    availability: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "terminal_discovery_receipt_sha256": terminal_sha256,
        "arms": {
            "NARRATIVE": {"status": "available", **ranking_records["NARRATIVE"]},
            "FIXED-O0": {"status": "available", **ranking_records["FIXED-O0"]},
            "ADAPTIVE": _unavailable_arm(terminal_sha256),
            "COMPOSITE": _unavailable_arm(terminal_sha256),
        },
    }
    availability_bytes = _pretty_bytes(availability)
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": topic_ids,
        "protected_topic_count": 0,
        "document_count": len(narrative),
        "rankings": ranking_records,
        "availability": _artifact_record(
            availability_bytes, 1, "availability.json"
        ),
        "source_bindings": bindings,
        "selection_reason_counts": {
            "NARRATIVE": dict(sorted(Counter(str(row["selection_reason"]) for row in narrative).items())),
            "FIXED-O0": dict(sorted(Counter(str(row["selection_reason"]) for row in fixed).items())),
        },
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "model_load_count": 0,
        "inference_count": 0,
        "external_cost_usd": 0.0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    for name, content in (
        ("narrative.jsonl", narrative_bytes),
        ("fixed_o0.jsonl", fixed_bytes),
        ("availability.json", availability_bytes),
        ("receipt.json", _pretty_bytes(receipt)),
    ):
        _exclusive_write(destination / name, content)
    return receipt


def _require_output_inventory(root: Path) -> None:
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise ValueError("ranking output inventory is unreadable") from exc
    if {entry.name for entry in entries} != _OUTPUT_NAMES or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("ranking output inventory is partial, unexpected, or unsafe")


def _verify_artifact(
    root: Path,
    binding: object,
    *,
    expected_path: str,
    label: str,
) -> bytes:
    if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
        raise ValueError(f"{label} artifact binding is invalid")
    try:
        content = (root / expected_path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} artifact is unreadable") from exc
    if binding.get("bytes") != len(content) or binding.get("sha256") != _sha256_bytes(content):
        raise ValueError(f"{label} artifact hash or byte count differs")
    return content


def _verify_ranking_rows(
    rows: Sequence[Mapping[str, object]], topic_ids: Sequence[str], label: str
) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    by_topic: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if set(row) != _REQUIRED_ROW_FIELDS:
            raise ValueError(f"{label} ranking row fields differ")
        topic_id = row.get("topic_id")
        document_id = row.get("document_id")
        score = row.get("score")
        scored_obligations = row.get("scored_obligations")
        if (
            row.get("schema_version") != ROW_SCHEMA_VERSION
            or not isinstance(topic_id, str)
            or topic_id not in topic_ids
            or not isinstance(document_id, str)
            or not document_id
            or (topic_id, document_id) in identities
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or row.get("primary_obligation") != row.get("score_scope")
            or not isinstance(scored_obligations, list)
            or row.get("score_scope") not in scored_obligations
            or any(not _is_sha256(row.get(name)) for name in (
                "document_sha256", "window_sha256", "query_sha256",
                "preflight_sha256", "preflight_windows_sha256",
            ))
        ):
            raise ValueError(f"{label} ranking row identity or provenance is invalid")
        identities.add((topic_id, document_id))
        by_topic[topic_id].append(row)
    if [topic for topic in topic_ids if topic in by_topic] != list(topic_ids):
        raise ValueError(f"{label} ranking topic coverage differs")
    for topic_id in topic_ids:
        local = by_topic[topic_id]
        if [row.get("topic_rank") for row in local] != list(range(1, len(local) + 1)):
            raise ValueError(f"{label} per-topic ranks are not exact")
    return identities


def _verify_source_bindings(bindings: Mapping[str, object]) -> dict[str, object] | None:
    if bindings.get("mode") != "authenticated_paths":
        return None
    path_fields = {
        "contract_dir": "contract_summary_sha256",
        "base_scores_dir": "base_score_receipt_sha256",
        "discovery_dir": "terminal_discovery_receipt_sha256",
    }
    for name, hash_name in path_fields.items():
        if not isinstance(bindings.get(name), str) or not _is_sha256(bindings.get(hash_name)):
            raise ValueError("authenticated ranking source binding is invalid")
    data = load_authenticated_ranking_data(
        contract_dir=Path(str(bindings["contract_dir"])),
        scores_dir=Path(str(bindings["base_scores_dir"])),
        discovery_dir=Path(str(bindings["discovery_dir"])),
    )
    observed = _source_bindings(data)
    if observed != dict(bindings):
        raise ValueError("authenticated ranking source receipt hash differs")
    return data


def verify_baseline_rankings(
    output_dir: Path, *, data: object | None = None
) -> dict[str, object]:
    """Authenticate output inventory, bindings, completeness, and deterministic rows."""

    root = Path(output_dir)
    _require_output_inventory(root)
    receipt = _read_json(root / "receipt.json", "ranking receipt")
    raw_topic_ids = receipt.get("topic_ids")
    if not isinstance(raw_topic_ids, list) or any(
        not isinstance(topic_id, str) for topic_id in raw_topic_ids
    ):
        raise ValueError("ranking receipt topic scope is invalid")
    topic_ids = list(raw_topic_ids)
    _reject_protected(topic_ids)
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("protected_topic_count") != 0
        or receipt.get("qrels_opened") is not False
        or any(receipt.get(name) != 0 for name in _ZERO_COUNTERS)
        or receipt.get("external_cost_usd") != 0.0
    ):
        raise ValueError("ranking receipt status or safety counters differ")
    rankings = receipt.get("rankings")
    if not isinstance(rankings, Mapping) or set(rankings) != set(_AVAILABLE_ARMS):
        raise ValueError("ranking receipt arm bindings differ")
    parsed: dict[str, list[dict[str, object]]] = {}
    identities: dict[str, set[tuple[str, str]]] = {}
    for arm in _AVAILABLE_ARMS:
        name = _RANKING_NAMES[arm]
        content = _verify_artifact(
            root, rankings.get(arm), expected_path=name, label=arm
        )
        parsed[arm] = _read_jsonl(root / name, arm)
        binding = rankings[arm]
        assert isinstance(binding, Mapping)
        if binding.get("rows") != len(parsed[arm]) or len(content.splitlines()) != len(parsed[arm]):
            raise ValueError(f"{arm} ranking row count differs")
        identities[arm] = _verify_ranking_rows(parsed[arm], topic_ids, arm)
    if identities["NARRATIVE"] != identities["FIXED-O0"]:
        raise ValueError("baseline ranking populations differ")
    if receipt.get("document_count") != len(identities["NARRATIVE"]):
        raise ValueError("ranking receipt document count differs")

    availability_content = _verify_artifact(
        root,
        receipt.get("availability"),
        expected_path="availability.json",
        label="availability",
    )
    availability = _read_json(root / "availability.json", "ranking availability")
    if availability_content != _pretty_bytes(availability):
        raise ValueError("ranking availability is not canonical JSON")
    bindings = receipt.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("ranking source bindings are missing")
    terminal_sha256 = bindings.get("terminal_discovery_receipt_sha256")
    arms = availability.get("arms")
    if (
        availability.get("schema_version") != SCHEMA_VERSION
        or availability.get("terminal_discovery_receipt_sha256") != terminal_sha256
        or not isinstance(arms, Mapping)
        or set(arms) != {*_AVAILABLE_ARMS, *_UNAVAILABLE_ARMS}
    ):
        raise ValueError("ranking arm availability differs")
    for arm in _AVAILABLE_ARMS:
        value = arms.get(arm)
        if not isinstance(value, Mapping) or dict(value) != {
            "status": "available", **dict(rankings[arm])  # type: ignore[arg-type]
        }:
            raise ValueError(f"{arm} availability binding differs")
    expected_unavailable = _unavailable_arm(str(terminal_sha256))
    for arm in _UNAVAILABLE_ARMS:
        value = arms.get(arm)
        if not isinstance(value, Mapping) or dict(value) != expected_unavailable:
            raise ValueError(f"{arm} must be explicitly unavailable")

    authenticated_data = _verify_source_bindings(bindings)
    comparison_data = data if data is not None else authenticated_data
    if comparison_data is not None:
        if _topic_ids(comparison_data) != topic_ids:
            raise ValueError("ranking topics differ from authenticated sources")
        expected = {
            "NARRATIVE": build_narrative_continuation(comparison_data),
            "FIXED-O0": build_fixed_o0_continuation(comparison_data),
        }
        if any(parsed[arm] != expected[arm] for arm in _AVAILABLE_ARMS):
            raise ValueError("ranking rows differ from deterministic source reconstruction")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build both complete baseline rankings")
    build.add_argument("--contract", type=Path, required=True)
    build.add_argument("--scores", type=Path, required=True)
    build.add_argument("--discovery", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="authenticate both baseline rankings")
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        if _path_present(args.output):
            raise FileExistsError(f"create-only ranking output exists: {args.output}")
        data = load_authenticated_ranking_data(
            contract_dir=args.contract,
            scores_dir=args.scores,
            discovery_dir=args.discovery,
        )
        receipt = build_rankings(data, output_dir=args.output)
        print(
            "status=complete "
            f"documents={receipt['document_count']} "
            "arms=NARRATIVE,FIXED-O0 adaptive=unavailable composite=unavailable "
            "network=false qrels_opened=false external_cost=$0"
        )
    else:
        receipt = verify_baseline_rankings(args.output)
        print(
            "status=verified "
            f"documents={receipt['document_count']} "
            "arms=NARRATIVE,FIXED-O0 adaptive=unavailable composite=unavailable "
            "network=false qrels_opened=false external_cost=$0"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
