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
CASE_REGISTRY_VERSION = "semantic_anchor_case_registry_v1"

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

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "superpowers"
    / "det_sparse_v4_contract_artifacts"
)
ARTIFACT_MANIFEST = ARTIFACT_DIR / "semantic_anchor_artifact_manifest_v1.json"

SELECT_POSITIONS = ("leading", "middle", "trailing")
SELECT_ANCHOR_CLASSES = (
    "multiword_entity_topic",
    "punctuation_unicode_phrase",
    "comparison_pair",
)
SELECT_CHILD_REFERENCE_STYLES = ("pronoun", "ellipsis_generic")
ABSTAIN_REASONS = (
    "coequal_disjoint_subjects",
    "generic_boilerplate_no_referent",
    "referent_only_outside_u1",
    "only_one_eligible_anchor_term",
    "all_ranges_violate_span_or_occurrence_cap",
    "adversarial_instruction_no_unambiguous_referent",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_no_duplicates(path: Path) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def load_jsonl_no_duplicates(path: Path) -> tuple[object, ...]:
    records: list[object] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"blank JSONL line in {path}: {line_number}")

        def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(
                        f"duplicate JSON key in {path}:{line_number}: {key}"
                    )
                result[key] = value
            return result

        records.append(json.loads(line, object_pairs_hook=unique_object))
    return tuple(records)


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


def expected_case001_response_schema() -> dict[str, object]:
    return model_response_schema("synthetic-case-001", u1_token_count=7)


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


def validate_case_registry(registry: Mapping[str, object]) -> tuple[str, ...]:
    if registry.get("schema_version") != CASE_REGISTRY_VERSION:
        raise ValueError("case registry schema_version mismatch")
    cases = registry.get("cases")
    if not isinstance(cases, list):
        raise ValueError("case registry cases must be a list")
    if len(cases) != 24:
        raise ValueError(f"case registry must contain 24 cases, got {len(cases)}")

    case_ids: list[str] = []
    select_cells: set[tuple[str, str, str]] = set()
    abstain_reasons: list[str] = []
    for index, value in enumerate(cases, start=1):
        case = _as_mapping(value, "case registry entry")
        case_id = case.get("case_id")
        expected_case_id = f"synthetic-case-{index:03d}"
        if case_id != expected_case_id:
            raise ValueError(f"case registry expected {expected_case_id}, got {case_id}")
        case_ids.append(expected_case_id)
        decision = case.get("decision")
        if decision == "select":
            position = _required_enum(case, "u1_anchor_position", SELECT_POSITIONS)
            anchor_class = _required_enum(case, "anchor_class", SELECT_ANCHOR_CLASSES)
            reference_style = _required_enum(
                case, "child_reference_style", SELECT_CHILD_REFERENCE_STYLES
            )
            select_cells.add((position, anchor_class, reference_style))
            if "abstain_reason" in case:
                raise ValueError(f"select case must not have abstain_reason: {case_id}")
        elif decision == "abstain":
            reason = _required_enum(case, "abstain_reason", ABSTAIN_REASONS)
            abstain_reasons.append(reason)
            for forbidden in (
                "u1_anchor_position",
                "anchor_class",
                "child_reference_style",
            ):
                if forbidden in case:
                    raise ValueError(f"abstain case must not have {forbidden}: {case_id}")
        else:
            raise ValueError(f"case decision must be select or abstain: {case_id}")

    expected_cells = {
        (position, anchor_class, reference_style)
        for position in SELECT_POSITIONS
        for anchor_class in SELECT_ANCHOR_CLASSES
        for reference_style in SELECT_CHILD_REFERENCE_STYLES
    }
    if select_cells != expected_cells:
        raise ValueError("select registry must be exact 3x3x2 Cartesian product")
    if tuple(abstain_reasons) != ABSTAIN_REASONS:
        raise ValueError("abstain registry reasons drifted")
    return tuple(case_ids)


def validate_artifact_bundle(
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, str]:
    manifest_path = artifact_dir / "semantic_anchor_artifact_manifest_v1.json"
    manifest = _as_mapping(load_json_no_duplicates(manifest_path), "artifact manifest")
    if manifest.get("schema_version") != "semantic_anchor_artifact_manifest_v1":
        raise ValueError("artifact manifest schema_version mismatch")
    if manifest.get("artifact_set_status") != "offline_24_fixture_set_not_inference_authorizing":
        raise ValueError("artifact manifest must remain offline non-inference fixture set")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact manifest must list artifacts")

    seen: set[str] = set()
    runner_files: list[str] = []
    scorer_files: list[str] = []
    hashes: dict[str, str] = {}
    for item in artifacts:
        entry = _as_mapping(item, "artifact entry")
        path_value = entry.get("path")
        visibility = entry.get("visibility")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("artifact entry path must be nonempty string")
        if "/" in path_value or path_value.startswith("."):
            raise ValueError(f"artifact path must be local basename: {path_value}")
        if path_value in seen:
            raise ValueError(f"duplicate artifact path: {path_value}")
        seen.add(path_value)
        if visibility == "runner":
            runner_files.append(path_value)
        elif visibility == "scorer":
            scorer_files.append(path_value)
        elif visibility != "reviewer":
            raise ValueError(f"unknown artifact visibility for {path_value}: {visibility}")
        path = artifact_dir / path_value
        if not path.is_file():
            raise ValueError(f"artifact missing: {path_value}")
        hashes[path_value] = sha256_file(path)

    expected = set(runner_visible_artifacts()).union(scorer_only_artifacts())
    missing_expected = expected.difference(runner_files).difference(scorer_files)
    if missing_expected:
        raise ValueError(f"manifest missing expected artifacts: {sorted(missing_expected)}")
    validate_runner_gold_separation(runner_files, scorer_files)
    _validate_artifact_cross_file_consistency(artifact_dir)
    return hashes


def runner_visible_artifacts() -> tuple[str, ...]:
    return (
        "lucene_whole_unit_offsets_request_v1.schema.json",
        "lucene_whole_unit_offsets_response_v1.schema.json",
        "lucene_whole_unit_offsets_health_v1.schema.json",
        "lucene_whole_unit_offsets_error_v1.schema.json",
        "semantic_anchor_response_v1.case001.schema.json",
        "semantic_anchor_case_registry_v1.json",
        "semantic_anchor_synthetic_corpus_v1.json",
        "semantic_anchor_case_order_v1.json",
        "semantic_anchor_prompt_v1.system.txt",
        "semantic_anchor_prompt_v1.user_template.json",
        "semantic_anchor_request_fixture.case001.json",
        "semantic_anchor_request_fixtures_v1.jsonl",
        "semantic_anchor_terminal_receipt_v1.schema.json",
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


def _validate_artifact_cross_file_consistency(artifact_dir: Path) -> None:
    response_schema = load_json_no_duplicates(
        artifact_dir / "semantic_anchor_response_v1.case001.schema.json"
    )
    if response_schema != expected_case001_response_schema():
        raise ValueError("case001 response schema fixture does not match generator")

    system_prompt = (artifact_dir / "semantic_anchor_prompt_v1.system.txt").read_text(
        encoding="utf-8"
    ).rstrip("\n")

    case_order = _as_mapping(
        load_json_no_duplicates(artifact_dir / "semantic_anchor_case_order_v1.json"),
        "case order",
    )
    registry = _as_mapping(
        load_json_no_duplicates(artifact_dir / "semantic_anchor_case_registry_v1.json"),
        "case registry",
    )
    registry_order = validate_case_registry(registry)
    if case_order.get("case_order") != list(registry_order):
        raise ValueError("case order fixture mismatch")
    if case_order.get("smoke_case_id") != "synthetic-case-001":
        raise ValueError("smoke case fixture mismatch")

    corpus = _as_mapping(
        load_json_no_duplicates(artifact_dir / "semantic_anchor_synthetic_corpus_v1.json"),
        "synthetic corpus",
    )
    cases = corpus.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("corpus must contain exactly 24 cases")
    corpus_by_id = {
        _as_mapping(case, "synthetic case").get("case_id"): _as_mapping(
            case, "synthetic case"
        )
        for case in cases
    }
    if tuple(corpus_by_id) != registry_order:
        raise ValueError("corpus case order differs from registry")

    gold = _as_mapping(
        load_json_no_duplicates(artifact_dir / "semantic_anchor_gold_labels_v1.json"),
        "gold labels",
    )
    gold_cases = gold.get("cases")
    if not isinstance(gold_cases, list) or len(gold_cases) != 24:
        raise ValueError("gold must contain exactly 24 cases")
    gold_by_id = {
        _as_mapping(case, "gold case").get("case_id"): _as_mapping(case, "gold case")
        for case in gold_cases
    }
    if tuple(gold_by_id) != registry_order:
        raise ValueError("gold case order differs from registry")

    request_records = load_jsonl_no_duplicates(
        artifact_dir / "semantic_anchor_request_fixtures_v1.jsonl"
    )
    if len(request_records) != 24:
        raise ValueError("request fixture JSONL must contain exactly 24 records")
    request_by_id: dict[str, Mapping[str, object]] = {}
    for record in request_records:
        entry = _as_mapping(record, "request fixture record")
        case_id = entry.get("case_id")
        request = _as_mapping(entry.get("request"), "request fixture request")
        if not isinstance(case_id, str):
            raise ValueError("request fixture case_id must be string")
        request_by_id[case_id] = request
    if tuple(request_by_id) != registry_order:
        raise ValueError("request fixture order differs from registry")

    case001_request = _as_mapping(
        load_json_no_duplicates(artifact_dir / "semantic_anchor_request_fixture.case001.json"),
        "case001 request fixture",
    )
    if case001_request != request_by_id["synthetic-case-001"]:
        raise ValueError("case001 single request fixture differs from JSONL")

    registry_cases = _as_mapping(registry, "case registry").get("cases")
    assert isinstance(registry_cases, list)
    registry_by_id = {
        _as_mapping(case, "case registry entry").get("case_id"): _as_mapping(
            case, "case registry entry"
        )
        for case in registry_cases
    }
    for case_id in registry_order:
        case = _as_mapping(corpus_by_id[case_id], "synthetic case")
        gold_case = _as_mapping(gold_by_id[case_id], "gold case")
        request = _as_mapping(request_by_id[case_id], "request fixture")
        registry_case = _as_mapping(registry_by_id[case_id], "case registry entry")
        _validate_one_case_fixture(
            case_id=case_id,
            case=case,
            gold_case=gold_case,
            request=request,
            registry_case=registry_case,
            system_prompt=system_prompt,
        )


def _validate_one_case_fixture(
    *,
    case_id: str,
    case: Mapping[str, object],
    gold_case: Mapping[str, object],
    request: Mapping[str, object],
    registry_case: Mapping[str, object],
    system_prompt: str,
) -> None:
    if case.get("schema_version") != "semantic_anchor_synthetic_case_v1":
        raise ValueError(f"case schema_version mismatch: {case_id}")
    if gold_case.get("schema_version") != "semantic_anchor_gold_label_v1":
        raise ValueError(f"gold schema_version mismatch: {case_id}")
    if case.get("case_id") != case_id or gold_case.get("case_id") != case_id:
        raise ValueError(f"case/gold ID mismatch: {case_id}")
    token_tape = case.get("token_tape")
    if not isinstance(token_tape, list) or len(token_tape) != 7:
        raise ValueError(f"case must have seven token records: {case_id}")
    if [record.get("token_id") for record in token_tape if isinstance(record, dict)] != list(range(7)):
        raise ValueError(f"token IDs must be 0..6: {case_id}")
    unit_boundaries = case.get("unit_boundaries")
    if unit_boundaries != [
        {"unit_id": "u01", "start_token": 0, "end_token": 5},
        {"unit_id": "u02", "start_token": 5, "end_token": 6},
        {"unit_id": "u03", "start_token": 6, "end_token": 7},
    ]:
        raise ValueError(f"unit boundaries drifted: {case_id}")

    decision = registry_case.get("decision")
    if gold_case.get("decision") != decision:
        raise ValueError(f"gold decision differs from registry: {case_id}")
    ranges = gold_case.get("acceptable_ranges")
    if not isinstance(ranges, list):
        raise ValueError(f"acceptable_ranges must be list: {case_id}")
    if decision == "select":
        if len(ranges) != 1:
            raise ValueError(f"select case must have one accepted range: {case_id}")
        span = _as_mapping(ranges[0], "accepted range")
        validate_model_response_shape(
            {
                "schema_version": MODEL_RESPONSE_SCHEMA_VERSION,
                "case_id": case_id,
                "decision": "select",
                "start_token": span.get("start_token"),
                "end_token": span.get("end_token"),
            },
            case_id=case_id,
            u1_token_count=7,
        )
        if span["start_token"] < 0 or span["end_token"] > 5:
            raise ValueError(f"select range must be wholly inside U1: {case_id}")
        evidence = case.get("analyzer_evidence")
        if not isinstance(evidence, list) or len(evidence) < 2:
            raise ValueError(f"select case must expose at least two analyzer terms: {case_id}")
    else:
        if ranges:
            raise ValueError(f"abstain case must not have accepted ranges: {case_id}")
        if gold_case.get("abstain_reason") != registry_case.get("abstain_reason"):
            raise ValueError(f"abstain reason differs from registry: {case_id}")

    if request.get("model") != "gpt-oss-local":
        raise ValueError(f"request fixture must use served local model alias: {case_id}")
    if request.get("max_tokens") != 512:
        raise ValueError(f"request fixture max_tokens mismatch: {case_id}")
    if request.get("temperature") != 1.0 or request.get("seed") != 0:
        raise ValueError(f"request fixture sampling mismatch: {case_id}")
    if request.get("reasoning_effort") != "low":
        raise ValueError(f"request fixture reasoning_effort mismatch: {case_id}")
    response_format = _as_mapping(request.get("response_format"), "response_format")
    if response_format.get("type") != "json_schema":
        raise ValueError(f"request fixture response_format type mismatch: {case_id}")
    json_schema = _as_mapping(response_format.get("json_schema"), "json_schema")
    if json_schema.get("strict") is not True:
        raise ValueError(f"request fixture must set strict json_schema: {case_id}")
    expected_schema = model_response_schema(case_id, u1_token_count=7)
    if json_schema.get("schema") != expected_schema:
        raise ValueError(f"request fixture embeds different response schema: {case_id}")

    messages = request.get("messages")
    if not isinstance(messages, list) or [m.get("role") for m in messages if isinstance(m, dict)] != [
        "system",
        "user",
    ]:
        raise ValueError(f"request fixture must have exact system,user messages: {case_id}")
    if messages[0].get("content") != system_prompt:
        raise ValueError(f"request fixture system prompt drift: {case_id}")
    user_payload = json.loads(messages[1].get("content"), object_pairs_hook=dict)
    for key in ("case_id", "narrative", "token_tape", "unit_boundaries", "analyzer_evidence"):
        if user_payload.get(key) != case.get(key):
            raise ValueError(f"request fixture {key} differs from corpus: {case_id}")


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _required_enum(
    value: Mapping[str, object], key: str, allowed: Sequence[str]
) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or candidate not in allowed:
        raise ValueError(f"{key} must be one of {tuple(allowed)}")
    return candidate


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
