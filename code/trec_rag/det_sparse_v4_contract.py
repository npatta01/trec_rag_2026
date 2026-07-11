"""Offline contracts for deterministic sparse v4 synthetic qualification.

This module is intentionally topic-free.  It provides static schemas, canonical
byte helpers, fixture registries, and fail-closed validators that can be tested
without opening topics, qrels, retrieval caches, reranker state, model files, or
network endpoints.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "det_sparse_exact_span_synthetic_v4"
SCHEMA_VERSION = "det_sparse_v4_contract_v1"
MODEL_RESPONSE_SCHEMA_VERSION = "semantic_anchor_response_v1"
OFFSET_REQUEST_SCHEMA_VERSION = "lucene_whole_unit_offsets_request_v1"
OFFSET_RESPONSE_SCHEMA_VERSION = "lucene_whole_unit_offsets_response_v1"
SYNTHETIC_CORPUS_VERSION = "semantic_anchor_synthetic_corpus_v1"
QUALIFICATION_LEDGER_VERSION = "semantic_anchor_qualification_ledger_v1"

KNOWN_FIVE_TOPIC_IDS = ("144", "213", "224", "407", "515")
V1_TOPIC_IDS = ("200", "225", "707", "897")
V2_TOPIC_IDS = ("37", "84", "161", "300")
V3_TOPIC_IDS = ("14", "31", "58", "72", "219", "233", "273", "477", "499")
DENIED_TOPIC_IDS = tuple(
    sorted((*KNOWN_FIVE_TOPIC_IDS, *V1_TOPIC_IDS, *V2_TOPIC_IDS, *V3_TOPIC_IDS), key=int)
)

DENIED_IMPORTS = frozenset(
    {
        "trec_rag.topics",
        "trec_rag.query_planner",
        "trec_rag.deterministic_sparse",
        "trec_rag.deterministic_sparse_v2",
        "trec_rag.deterministic_sparse_v3",
        "trec_rag.retrievers",
        "trec_rag.ranking",
        "trec_rag.remote_client",
        "trec_rag.remote_pyserini",
        "trec_rag.rerank_score_cache",
        "trec_rag.evaluation",
    }
)

DENIED_PATH_FRAGMENTS = (
    "rag25-topics-dev.tsv",
    "research-rubrics",
    "qrels",
    "cache/retrieval",
    "cache/reranker",
)


class TerminalState(str, Enum):
    PREFLIGHT_NO_GO = "preflight_no_go"
    PRE_DISPATCH_NO_GO = "pre_dispatch_no_go"
    FIRST_CASE_NO_GO = "first_case_no_go"
    TRANSPORT_NO_BODY_NO_GO = "transport_no_body_no_go"
    PREFIX_INTEGRITY_NO_GO = "prefix_integrity_no_go"
    COMPLETED_QUALIFICATION_NO_GO = "completed_qualification_no_go"
    COMPLETED_SYNTHETIC_GO = "completed_synthetic_go"
    INTERRUPTED_INCOMPLETE = "interrupted_incomplete"


@dataclass(frozen=True)
class DecisionBoundaryCase:
    decision: str
    start_label: str
    start_token: int
    end_label: str
    end_token: int
    valid_reason: str | None

    @property
    def is_valid(self) -> bool:
        return self.valid_reason is not None


@dataclass(frozen=True)
class ImportAuditIssue:
    path: str
    module: str
    line: int


def canonical_json_bytes(value: object) -> bytes:
    """Return the committed compact UTF-8 JSON byte representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def build_offset_request(text: str) -> tuple[dict[str, str], bytes, str]:
    request = {
        "schema_version": OFFSET_REQUEST_SCHEMA_VERSION,
        "text": text,
    }
    body = canonical_json_bytes(request)
    return request, body, sha256_bytes(body)


def model_response_schema(case_id: str, *, u1_token_count: int) -> dict[str, object]:
    if not case_id:
        raise ValueError("case_id must be nonempty")
    if u1_token_count < 1:
        raise ValueError("u1_token_count must be positive")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "case_id",
            "decision",
            "start_token",
            "end_token",
        ],
        "properties": {
            "schema_version": {"const": MODEL_RESPONSE_SCHEMA_VERSION, "type": "string"},
            "case_id": {"const": case_id, "type": "string"},
            "decision": {"enum": ["select", "abstain"], "type": "string"},
            "start_token": {
                "type": "integer",
                "minimum": -1,
                "maximum": u1_token_count,
            },
            "end_token": {
                "type": "integer",
                "minimum": -1,
                "maximum": u1_token_count,
            },
        },
    }


def validate_model_response_shape(
    response: Mapping[str, object], *, case_id: str, u1_token_count: int
) -> None:
    expected_keys = {"schema_version", "case_id", "decision", "start_token", "end_token"}
    keys = set(response)
    if keys != expected_keys:
        raise ValueError(f"response keys mismatch: missing={expected_keys - keys} extra={keys - expected_keys}")
    if response["schema_version"] != MODEL_RESPONSE_SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    if response["case_id"] != case_id:
        raise ValueError("case_id mismatch")
    decision = response["decision"]
    if decision not in {"select", "abstain"}:
        raise ValueError("decision must be select or abstain")
    start = _strict_int(response["start_token"], "start_token")
    end = _strict_int(response["end_token"], "end_token")
    if start < -1 or end < -1 or start > u1_token_count or end > u1_token_count:
        raise ValueError("range outside schema bounds")
    if decision == "abstain":
        if (start, end) != (-1, -1):
            raise ValueError("abstain requires -1/-1 sentinel")
        return
    if start < 0 or end <= start:
        raise ValueError("select requires a nonempty nonnegative range")


def decision_boundary_grid(*, u1_token_count: int) -> tuple[DecisionBoundaryCase, ...]:
    if u1_token_count < 3:
        raise ValueError("u1_token_count must be at least 3 for distinct boundary fixtures")
    labels_to_values = (
        ("neg2", -2),
        ("sentinel", -1),
        ("zero", 0),
        ("one", 1),
        ("u1_last_token", u1_token_count - 1),
        ("u1_exclusive_end", u1_token_count),
    )
    values = [value for _, value in labels_to_values]
    if len(values) != len(set(values)):
        raise ValueError("boundary values must be distinct")

    cases: list[DecisionBoundaryCase] = []
    for decision in ("abstain", "select"):
        for start_label, start in labels_to_values:
            for end_label, end in labels_to_values:
                cases.append(
                    DecisionBoundaryCase(
                        decision=decision,
                        start_label=start_label,
                        start_token=start,
                        end_label=end_label,
                        end_token=end,
                        valid_reason=_boundary_valid_reason(
                            decision=decision,
                            start=start,
                            end=end,
                            u1_token_count=u1_token_count,
                        ),
                    )
                )
    return tuple(cases)


def validate_denied_topic_sets() -> None:
    groups = (KNOWN_FIVE_TOPIC_IDS, V1_TOPIC_IDS, V2_TOPIC_IDS, V3_TOPIC_IDS)
    seen: set[str] = set()
    for group in groups:
        if len(group) != len(set(group)):
            raise ValueError("duplicate topic ID within denied group")
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(f"denied topic groups overlap: {sorted(overlap)}")
        seen.update(group)
    if len(seen) != 22:
        raise ValueError(f"denied topic union must have size 22, got {len(seen)}")


def audit_imports(
    paths: Iterable[Path],
    *,
    denied_imports: Iterable[str] = DENIED_IMPORTS,
) -> tuple[ImportAuditIssue, ...]:
    denied = tuple(sorted(denied_imports))
    issues: list[ImportAuditIssue] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    matched = _matching_denied_import(alias.name, denied)
                    if matched:
                        issues.append(ImportAuditIssue(str(path), matched, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                matched = _matching_denied_import(module, denied)
                if matched:
                    issues.append(ImportAuditIssue(str(path), matched, node.lineno))
    return tuple(issues)


def audit_denied_path_fragments(text: str) -> tuple[str, ...]:
    return tuple(fragment for fragment in DENIED_PATH_FRAGMENTS if fragment in text)


def validate_terminal_receipt(receipt: Mapping[str, object]) -> None:
    state = TerminalState(str(receipt.get("terminal_state")))
    attempted = _strict_int(receipt.get("attempted_calls"), "attempted_calls")
    completed = _strict_int(receipt.get("completed_calls"), "completed_calls")
    raw_committed = _strict_int(receipt.get("raw_committed_calls"), "raw_committed_calls")
    if min(attempted, completed, raw_committed) < 0:
        raise ValueError("call counters must be nonnegative")
    if completed > attempted:
        raise ValueError("completed_calls cannot exceed attempted_calls")
    if raw_committed > completed:
        raise ValueError("raw_committed_calls cannot exceed completed_calls")
    if receipt.get("schema_version") != "semantic_anchor_terminal_receipt_v1":
        raise ValueError("terminal receipt schema_version mismatch")
    if not isinstance(receipt.get("artifact_sha256"), dict):
        raise ValueError("terminal receipt must bind artifact_sha256 object")

    gold_opened = bool(receipt.get("gold_opened", False))
    if state in {
        TerminalState.PREFLIGHT_NO_GO,
        TerminalState.PRE_DISPATCH_NO_GO,
        TerminalState.FIRST_CASE_NO_GO,
        TerminalState.TRANSPORT_NO_BODY_NO_GO,
        TerminalState.PREFIX_INTEGRITY_NO_GO,
        TerminalState.INTERRUPTED_INCOMPLETE,
    } and gold_opened:
        raise ValueError("gold must remain unopened for incomplete/mechanical states")
    if state == TerminalState.PREFLIGHT_NO_GO and (attempted, completed, raw_committed) != (0, 0, 0):
        raise ValueError("preflight_no_go must have zero calls")
    if state == TerminalState.PRE_DISPATCH_NO_GO and attempted != 0:
        raise ValueError("pre_dispatch_no_go must have zero attempted calls")
    if state == TerminalState.FIRST_CASE_NO_GO and attempted != 1:
        raise ValueError("first_case_no_go must have exactly one attempted call")
    if state == TerminalState.TRANSPORT_NO_BODY_NO_GO and attempted < 1:
        raise ValueError("transport_no_body_no_go requires a dispatch attempt")
    if state in {TerminalState.COMPLETED_QUALIFICATION_NO_GO, TerminalState.COMPLETED_SYNTHETIC_GO}:
        if (attempted, completed, raw_committed) != (24, 24, 24):
            raise ValueError("completed states require 24 sealed raw responses")
        if not gold_opened:
            raise ValueError("completed states require scorer/gold phase")


def validate_model_inventory(inventory: Mapping[str, object]) -> None:
    if inventory.get("repository") != "openai/gpt-oss-20b":
        raise ValueError("unexpected model repository")
    if inventory.get("revision") != "6cee5e81ee83917806bbde320786a8fb61efebee":
        raise ValueError("unexpected model revision")
    if inventory.get("quantization_method") != "mxfp4":
        raise ValueError("unexpected quantization method")
    if inventory.get("safetensors_index_total_size") != 13_761_264_768:
        raise ValueError("unexpected safetensors index total size")
    loaded = _string_tuple(inventory.get("loaded_shards"), "loaded_shards")
    if len(loaded) != 3 or len(set(loaded)) != 3:
        raise ValueError("model inventory must have exactly three unique loaded shards")
    denied = set(_string_tuple(inventory.get("denied_files"), "denied_files"))
    if "original/model.safetensors" not in denied:
        raise ValueError("original/model.safetensors must be denied or hidden from loader")
    if "original/model.safetensors" in loaded:
        raise ValueError("original/model.safetensors must not be loaded")
    for key in ("snapshot_path", "file_sha256", "loaded_files", "unloaded_files"):
        if key not in inventory:
            raise ValueError(f"model inventory missing {key}")


def runner_visible_artifacts() -> tuple[str, ...]:
    return (
        "semantic_anchor_synthetic_corpus_v1.json",
        "semantic_anchor_case_order_v1.json",
        "semantic_anchor_response_v1.schema.json",
        "semantic_anchor_prompt_v1.system.txt",
        "semantic_anchor_prompt_v1.user_template.json",
        "semantic_anchor_request_fixtures_v1.jsonl",
    )


def scorer_only_artifacts() -> tuple[str, ...]:
    return (
        "semantic_anchor_gold_labels_v1.json",
        "semantic_anchor_scorer_v1.schema.json",
        "semantic_anchor_reviewer_rubric_v1.md",
    )


def validate_runner_gold_separation(runner_files: Sequence[str], gold_files: Sequence[str]) -> None:
    overlap = set(runner_files).intersection(gold_files)
    if overlap:
        raise ValueError(f"runner/gold artifact overlap: {sorted(overlap)}")
    for value in runner_files:
        lowered = value.lower()
        if "gold" in lowered or "label" in lowered or "scorer" in lowered:
            raise ValueError(f"runner-visible artifact appears to expose gold: {value}")


def _boundary_valid_reason(
    *, decision: str, start: int, end: int, u1_token_count: int
) -> str | None:
    if decision == "abstain":
        return "valid_abstain_sentinel" if (start, end) == (-1, -1) else None
    if decision == "select" and 0 <= start < end <= u1_token_count:
        return "valid_select_range"
    return None


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _matching_denied_import(module: str, denied: Sequence[str]) -> str | None:
    for candidate in denied:
        if module == candidate or module.startswith(candidate + "."):
            return candidate
    return None


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{name} must contain nonempty strings")
        result.append(item)
    return tuple(result)
