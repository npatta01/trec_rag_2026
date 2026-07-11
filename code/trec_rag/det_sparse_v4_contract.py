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
OFFSET_CONTRACT_VERSION = "lucene_whole_unit_offsets_v1"
OFFSET_UNIT = "unicode_code_points"
SYNTHETIC_CORPUS_VERSION = "semantic_anchor_synthetic_corpus_v1"
QUALIFICATION_LEDGER_VERSION = "semantic_anchor_qualification_ledger_v1"
CASE_REGISTRY_VERSION = "semantic_anchor_case_registry_v1"
EXPECTED_MODEL_REPOSITORY = "openai/gpt-oss-20b"
EXPECTED_MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
EXPECTED_SERVED_MODEL = "gpt-oss-local"
EXPECTED_VLLM_VERSION = "0.24.0"
EXPECTED_XGRAMMAR_VERSION = "0.2.3"
EXPECTED_STRUCTURED_OUTPUTS_BACKEND = "xgrammar"

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
    ".cache/huggingface",
    "huggingface/hub",
    "models--openai--gpt-oss-20b",
    "openai/gpt-oss-20b/snapshots",
    "model.safetensors",
    ".safetensors",
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
REPLAY_MUTATION_IDS = (
    "mutate_request_body_byte",
    "mutate_raw_response_body_byte",
    "swap_case_order_ids",
    "mark_prefix_gold_opened",
    "change_response_schema_case_const",
    "add_gold_to_runner_manifest",
    "change_model_inventory_hash",
    "mark_dispatch_non_loopback",
)
REPLAY_FAILURE_CODES = (
    "request_hash_mismatch",
    "raw_response_hash_mismatch",
    "case_order_mismatch",
    "gold_opened_before_complete",
    "schema_constant_mismatch",
    "runner_gold_visibility_violation",
    "model_inventory_changed",
    "non_loopback_dispatch",
)
LEDGER_SCHEMA_FILES = (
    "semantic_anchor_reservation_v1.schema.json",
    "semantic_anchor_pre_dispatch_attestation_v1.schema.json",
    "semantic_anchor_dispatch_record_v1.schema.json",
    "semantic_anchor_raw_response_body_v1.schema.json",
    "semantic_anchor_transport_failure_v1.schema.json",
    "semantic_anchor_case_receipt_v1.schema.json",
    "semantic_anchor_run_manifest_v1.schema.json",
    "semantic_anchor_terminal_receipt_v1.schema.json",
)
RENDERER_ORACLE_IDS = (
    "oracle_select_leading_entity",
    "oracle_select_comparison_pair",
    "oracle_abstain_coequal_subjects",
)
SCORER_CLASSIFICATIONS = (
    "correct_select",
    "safe_abstain",
    "wrong_referent",
    "wrong_abstain",
    "mechanical_failure",
)
ALLOWED_IMPORT_ROOTS = (
    "ast",
    "dataclasses",
    "enum",
    "hashlib",
    "json",
    "pathlib",
    "typing",
)


class TerminalState(str, Enum):
    PREFLIGHT_NO_GO = "preflight_no_go"
    PRE_DISPATCH_NO_GO = "pre_dispatch_no_go"
    FIRST_CASE_NO_GO = "first_case_no_go"
    TRANSPORT_NO_BODY_NO_GO = "transport_no_body_no_go"
    PREFIX_INTEGRITY_NO_GO = "prefix_integrity_no_go"
    RAW_SEALED_PENDING_SCORER = "raw_sealed_pending_scorer"
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


def validate_offset_response_shape(
    response: Mapping[str, object],
    *,
    text: str,
    expected_fingerprint: Mapping[str, object] | None = None,
) -> None:
    expected_keys = {
        "schema_version",
        "text_sha256",
        "offset_unit",
        "fingerprint",
        "occurrences",
    }
    keys = set(response)
    if keys != expected_keys:
        raise ValueError(f"offset response keys mismatch: missing={expected_keys - keys} extra={keys - expected_keys}")
    if response["schema_version"] != OFFSET_RESPONSE_SCHEMA_VERSION:
        raise ValueError("offset response schema_version mismatch")
    if response["offset_unit"] != OFFSET_UNIT:
        raise ValueError("offset response offset_unit mismatch")
    if response["text_sha256"] != text_sha256(text):
        raise ValueError("offset response text_sha256 mismatch")

    fingerprint = _as_mapping(response["fingerprint"], "offset fingerprint")
    validate_offset_fingerprint(fingerprint)
    if expected_fingerprint is not None and fingerprint != dict(expected_fingerprint):
        raise ValueError("offset response fingerprint drift")

    occurrences = response["occurrences"]
    if not isinstance(occurrences, list):
        raise ValueError("offset response occurrences must be list")
    previous_start = -1
    previous_end = -1
    text_length = len(text)
    for expected_ordinal, raw_occurrence in enumerate(occurrences):
        occurrence = _as_mapping(raw_occurrence, "offset occurrence")
        occurrence_keys = set(occurrence)
        expected_occurrence_keys = {
            "ordinal",
            "term",
            "start_codepoint",
            "end_codepoint",
            "position_increment",
        }
        if occurrence_keys != expected_occurrence_keys:
            raise ValueError(
                "offset occurrence keys mismatch: "
                f"missing={expected_occurrence_keys - occurrence_keys} "
                f"extra={occurrence_keys - expected_occurrence_keys}"
            )
        ordinal = _strict_int(occurrence["ordinal"], "ordinal")
        if ordinal != expected_ordinal:
            raise ValueError("offset occurrence ordinal gap")
        term = occurrence["term"]
        if not isinstance(term, str) or not term:
            raise ValueError("offset occurrence term must be nonempty string")
        start = _strict_int(occurrence["start_codepoint"], "start_codepoint")
        end = _strict_int(occurrence["end_codepoint"], "end_codepoint")
        position_increment = _strict_int(
            occurrence["position_increment"], "position_increment"
        )
        if position_increment < 0:
            raise ValueError("offset occurrence position_increment must be nonnegative")
        if start < 0 or end < 0 or start >= end:
            raise ValueError("offset occurrence must have nonempty nonnegative span")
        if end > text_length:
            raise ValueError("offset occurrence span exceeds text length")
        if start < previous_start:
            raise ValueError("offset occurrence starts are nonmonotone")
        if end < previous_end:
            raise ValueError("offset occurrence ends are nonmonotone")
        previous_start = start
        previous_end = end


def validate_offset_fingerprint(fingerprint: Mapping[str, object]) -> None:
    expected_keys = {
        "legacy_term_chain_fingerprint_sha256",
        "offset_contract_version",
        "offset_server_class_sha256",
        "offset_lucene_jar_sha256",
        "offset_runtime_image_digest",
    }
    keys = set(fingerprint)
    if keys != expected_keys:
        raise ValueError(f"offset fingerprint keys mismatch: missing={expected_keys - keys} extra={keys - expected_keys}")
    if fingerprint["offset_contract_version"] != OFFSET_CONTRACT_VERSION:
        raise ValueError("offset fingerprint contract version mismatch")
    for key in (
        "legacy_term_chain_fingerprint_sha256",
        "offset_server_class_sha256",
        "offset_lucene_jar_sha256",
    ):
        value = fingerprint[key]
        if not isinstance(value, str) or not _is_hex_sha256(value):
            raise ValueError(f"offset fingerprint {key} must be lowercase sha256")
    digest = fingerprint["offset_runtime_image_digest"]
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or not _is_hex_sha256(digest.removeprefix("sha256:"))
    ):
        raise ValueError("offset fingerprint runtime image digest must be sha256 digest")


def validate_offset_health(health: Mapping[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "legacy_analyzer_port",
        "offset_analyzer_port",
    }
    keys = set(health)
    if keys != expected_keys:
        raise ValueError(f"offset health keys mismatch: missing={expected_keys - keys} extra={keys - expected_keys}")
    if health["schema_version"] != "lucene_whole_unit_offsets_health_v1":
        raise ValueError("offset health schema_version mismatch")
    if health["status"] != "ok":
        raise ValueError("offset health status must be ok")
    if health["legacy_analyzer_port"] != 18081:
        raise ValueError("offset health legacy analyzer port mismatch")
    if health["offset_analyzer_port"] != 18082:
        raise ValueError("offset health offset analyzer port mismatch")


def validate_offset_error(error: Mapping[str, object]) -> None:
    expected_keys = {"schema_version", "error_code", "message"}
    keys = set(error)
    if keys != expected_keys:
        raise ValueError(f"offset error keys mismatch: missing={expected_keys - keys} extra={keys - expected_keys}")
    if error["schema_version"] != "lucene_whole_unit_offsets_error_v1":
        raise ValueError("offset error schema_version mismatch")
    if error["error_code"] not in {
        "bad_json",
        "schema_mismatch",
        "invalid_utf8",
        "analyzer_failure",
    }:
        raise ValueError("offset error_code mismatch")
    if not isinstance(error["message"], str) or not error["message"]:
        raise ValueError("offset error message must be nonempty string")


def validate_schema_compiler_attestation(
    record: Mapping[str, object],
    *,
    request_identity: Mapping[str, object],
) -> None:
    """Validate captured per-request XGrammar compiler evidence before dispatch."""

    _require_exact_mapping_keys(
        record,
        "schema compiler attestation",
        {"schema_version", "compiler", "case_order_sha256", "cases"},
    )
    if record.get("schema_version") != "semantic_anchor_schema_compiler_attestation_v1":
        raise ValueError("schema compiler attestation schema_version mismatch")

    compiler = _as_mapping(record.get("compiler"), "schema compiler identity")
    _require_exact_mapping_keys(
        compiler,
        "schema compiler identity",
        {"vllm_version", "xgrammar_version", "structured_outputs_backend"},
    )
    if compiler.get("vllm_version") != EXPECTED_VLLM_VERSION:
        raise ValueError("schema compiler vLLM version mismatch")
    if compiler.get("xgrammar_version") != EXPECTED_XGRAMMAR_VERSION:
        raise ValueError("schema compiler XGrammar version mismatch")
    if compiler.get("structured_outputs_backend") != EXPECTED_STRUCTURED_OUTPUTS_BACKEND:
        raise ValueError("schema compiler backend must be xgrammar")

    expected_order_sha256 = request_identity.get("case_order_sha256")
    _validate_sha256_string(expected_order_sha256, "request_identity case_order_sha256")
    if record.get("case_order_sha256") != expected_order_sha256:
        raise ValueError("schema compiler case_order_sha256 mismatch")
    request_sha256 = _as_mapping(request_identity.get("request_sha256"), "request_sha256")
    expected_schema_sha256 = _as_mapping(
        request_identity.get("schema_sha256"), "schema_sha256"
    )
    expected_case_order = tuple(request_sha256)
    if len(expected_case_order) != 24:
        raise ValueError("schema compiler request identity must contain 24 cases")
    if tuple(expected_schema_sha256) != expected_case_order:
        raise ValueError("schema compiler schema identity order mismatch")

    cases = record.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_case_order):
        raise ValueError("schema compiler cases must match request identity count")
    observed_order: list[str] = []
    for row in cases:
        case = _as_mapping(row, "schema compiler case")
        _require_exact_mapping_keys(
            case,
            "schema compiler case",
            {"case_id", "request_sha256", "schema_sha256", "xgrammar_strict"},
        )
        case_id = _validate_case_id(case.get("case_id"), "schema compiler case_id")
        observed_order.append(case_id)
        expected_request_sha256 = _validate_sha256_string(
            request_sha256.get(case_id), f"request_identity request_sha256[{case_id}]"
        )
        if case.get("request_sha256") != expected_request_sha256:
            raise ValueError("schema compiler request_sha256 mismatch")
        expected_schema_hash = _validate_sha256_string(
            expected_schema_sha256.get(case_id),
            f"request_identity schema_sha256[{case_id}]",
        )
        if case.get("schema_sha256") != expected_schema_hash:
            raise ValueError("schema compiler schema_sha256 mismatch")
        if case.get("xgrammar_strict") != "pass":
            raise ValueError("schema compiler xgrammar_strict must pass")
    if tuple(observed_order) != expected_case_order:
        raise ValueError("schema compiler case order mismatch")


def validate_live_model_runtime_attestation(
    record: Mapping[str, object],
    *,
    expected_model_inventory_sha256: str | None = None,
) -> None:
    """Validate captured local runtime identity without making a model call."""

    _require_exact_mapping_keys(
        record,
        "live model runtime attestation",
        {
            "schema_version",
            "attestation_id",
            "served_model",
            "repository",
            "revision",
            "vllm_version",
            "xgrammar_version",
            "structured_outputs_backend",
            "loopback_only",
            "egress_denied",
            "read_only_model_mount",
            "model_inventory_sha256",
        },
    )
    if record.get("schema_version") != "semantic_anchor_live_model_runtime_attestation_v1":
        raise ValueError("live model runtime attestation schema_version mismatch")
    _require_nonempty_string(record.get("attestation_id"), "attestation_id")
    if record.get("served_model") != EXPECTED_SERVED_MODEL:
        raise ValueError("live model runtime served_model mismatch")
    if record.get("repository") != EXPECTED_MODEL_REPOSITORY:
        raise ValueError("live model runtime repository mismatch")
    if record.get("revision") != EXPECTED_MODEL_REVISION:
        raise ValueError("live model runtime revision mismatch")
    if record.get("vllm_version") != EXPECTED_VLLM_VERSION:
        raise ValueError("live model runtime vLLM version mismatch")
    if record.get("xgrammar_version") != EXPECTED_XGRAMMAR_VERSION:
        raise ValueError("live model runtime XGrammar version mismatch")
    if record.get("structured_outputs_backend") != EXPECTED_STRUCTURED_OUTPUTS_BACKEND:
        raise ValueError("live model runtime backend must be xgrammar")
    if record.get("loopback_only") is not True:
        raise ValueError("live model runtime must be loopback_only")
    if record.get("egress_denied") is not True:
        raise ValueError("live model runtime must deny egress")
    if record.get("read_only_model_mount") is not True:
        raise ValueError("live model runtime requires read-only model mount")
    inventory_sha256 = _validate_sha256_string(
        record.get("model_inventory_sha256"), "model_inventory_sha256"
    )
    if (
        expected_model_inventory_sha256 is not None
        and inventory_sha256 != expected_model_inventory_sha256
    ):
        raise ValueError("live model runtime model inventory hash mismatch")


def pre_dispatch_attestation_from_live_runtime(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Project a strict live-runtime record into the ledger attestation shape."""

    validate_live_model_runtime_attestation(record)
    attestation = {
        "schema_version": "semantic_anchor_pre_dispatch_attestation_v1",
        "attestation_id": record["attestation_id"],
        "served_model": record["served_model"],
        "egress_denied": record["egress_denied"],
        "read_only_model_mount": record["read_only_model_mount"],
        "model_inventory_sha256": record["model_inventory_sha256"],
    }
    validate_pre_dispatch_attestation(attestation)
    return attestation


def validate_live_attestation_bundle(
    bundle: Mapping[str, object],
    *,
    request_identity: Mapping[str, object],
    expected_model_inventory_sha256: str | None = None,
) -> None:
    """Validate the local evidence bundle required before opening dispatch."""

    _require_exact_mapping_keys(
        bundle,
        "live attestation bundle",
        {"schema_version", "offset_health", "schema_compiler", "model_runtime"},
    )
    if bundle.get("schema_version") != "semantic_anchor_live_attestation_bundle_v1":
        raise ValueError("live attestation bundle schema_version mismatch")
    validate_offset_health(_as_mapping(bundle.get("offset_health"), "offset_health"))
    validate_schema_compiler_attestation(
        _as_mapping(bundle.get("schema_compiler"), "schema_compiler"),
        request_identity=request_identity,
    )
    validate_live_model_runtime_attestation(
        _as_mapping(bundle.get("model_runtime"), "model_runtime"),
        expected_model_inventory_sha256=expected_model_inventory_sha256,
    )


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


def extract_model_response_from_chat_completion(
    payload: Mapping[str, object], *, case_id: str, u1_token_count: int
) -> dict[str, object]:
    """Extract and validate the strict assistant JSON object from a raw response."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("chat completion must contain exactly one choice")
    choice = _as_mapping(choices[0], "chat completion choice")
    if choice.get("finish_reason") != "stop":
        raise ValueError("chat completion finish_reason must be stop")
    message = _as_mapping(choice.get("message"), "chat completion message")
    if message.get("role") != "assistant":
        raise ValueError("chat completion message role must be assistant")
    for forbidden in ("tool_calls", "function_call", "refusal"):
        if forbidden in message and message.get(forbidden) not in (None, [], ""):
            raise ValueError(f"chat completion message must not contain {forbidden}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("chat completion assistant content must be a nonempty string")
    if content != content.strip():
        raise ValueError("chat completion assistant content must not have outer whitespace")
    response = _loads_object_no_duplicates(content, "assistant content")
    validate_model_response_shape(response, case_id=case_id, u1_token_count=u1_token_count)
    return dict(response)


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
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        imported = f"{module}.*" if module else "*"
                    elif module:
                        imported = f"{module}.{alias.name}"
                    else:
                        imported = alias.name
                    matched = _matching_denied_import(imported, denied)
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
        TerminalState.RAW_SEALED_PENDING_SCORER,
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
    if state == TerminalState.RAW_SEALED_PENDING_SCORER:
        if (attempted, completed, raw_committed) != (24, 24, 24):
            raise ValueError("raw_sealed_pending_scorer requires 24 sealed raw responses")
    if state in {TerminalState.COMPLETED_QUALIFICATION_NO_GO, TerminalState.COMPLETED_SYNTHETIC_GO}:
        if (attempted, completed, raw_committed) != (24, 24, 24):
            raise ValueError("completed states require 24 sealed raw responses")
        if not gold_opened:
            raise ValueError("completed states require scorer/gold phase")


def validate_run_manifest(manifest: Mapping[str, object]) -> None:
    _require_exact_mapping_keys(
        manifest,
        "run manifest",
        {"schema_version", "case_count", "artifact_sha256", "terminal_receipt_path"},
    )
    if manifest.get("schema_version") != "semantic_anchor_run_manifest_v1":
        raise ValueError("run manifest schema_version mismatch")
    if manifest.get("case_count") != 24:
        raise ValueError("run manifest case_count must be 24")
    if manifest.get("terminal_receipt_path") != "terminal_receipt.json":
        raise ValueError("run manifest terminal_receipt_path mismatch")
    _validate_sha256_mapping(manifest.get("artifact_sha256"), "run manifest artifact_sha256")


def validate_reservation(record: Mapping[str, object]) -> None:
    _require_exact_mapping_keys(
        record,
        "reservation",
        {"schema_version", "run_id", "case_id", "request_sha256", "create_only"},
    )
    if record.get("schema_version") != "semantic_anchor_reservation_v1":
        raise ValueError("reservation schema_version mismatch")
    _require_nonempty_string(record.get("run_id"), "run_id")
    _validate_case_id(record.get("case_id"), "reservation case_id")
    _validate_sha256_string(record.get("request_sha256"), "reservation request_sha256")
    if record.get("create_only") is not True:
        raise ValueError("reservation must be create_only")


def validate_pre_dispatch_attestation(record: Mapping[str, object]) -> None:
    _require_exact_mapping_keys(
        record,
        "pre-dispatch attestation",
        {
            "schema_version",
            "attestation_id",
            "served_model",
            "egress_denied",
            "read_only_model_mount",
            "model_inventory_sha256",
        },
    )
    if record.get("schema_version") != "semantic_anchor_pre_dispatch_attestation_v1":
        raise ValueError("pre-dispatch attestation schema_version mismatch")
    _require_nonempty_string(record.get("attestation_id"), "attestation_id")
    if record.get("served_model") != "gpt-oss-local":
        raise ValueError("pre-dispatch attestation served_model mismatch")
    if record.get("egress_denied") is not True:
        raise ValueError("pre-dispatch attestation must deny egress")
    if record.get("read_only_model_mount") is not True:
        raise ValueError("pre-dispatch attestation requires read-only model mount")
    _validate_sha256_string(
        record.get("model_inventory_sha256"), "model_inventory_sha256"
    )


def validate_dispatch_record(record: Mapping[str, object]) -> None:
    _require_exact_mapping_keys(
        record,
        "dispatch record",
        {
            "schema_version",
            "case_id",
            "request_bytes_sha256",
            "loopback_only",
            "dispatch_counted",
        },
    )
    if record.get("schema_version") != "semantic_anchor_dispatch_record_v1":
        raise ValueError("dispatch record schema_version mismatch")
    _validate_case_id(record.get("case_id"), "dispatch case_id")
    _validate_sha256_string(record.get("request_bytes_sha256"), "request_bytes_sha256")
    if record.get("loopback_only") is not True:
        raise ValueError("dispatch record must be loopback_only")
    if record.get("dispatch_counted") is not True:
        raise ValueError("dispatch record must be dispatch_counted")


def validate_raw_response_record(record: Mapping[str, object]) -> None:
    _require_exact_mapping_keys(
        record,
        "raw response record",
        {
            "schema_version",
            "case_id",
            "http_status",
            "finish_reason",
            "served_model",
            "body_size_bytes",
            "body_sha256",
        },
    )
    if record.get("schema_version") != "semantic_anchor_raw_response_body_v1":
        raise ValueError("raw response schema_version mismatch")
    _validate_case_id(record.get("case_id"), "raw response case_id")
    if record.get("http_status") != 200:
        raise ValueError("raw response http_status must be 200")
    if record.get("finish_reason") != "stop":
        raise ValueError("raw response finish_reason must be stop")
    if record.get("served_model") != "gpt-oss-local":
        raise ValueError("raw response served_model mismatch")
    if _strict_int(record.get("body_size_bytes"), "body_size_bytes") < 1:
        raise ValueError("raw response body_size_bytes must be positive")
    _validate_sha256_string(record.get("body_sha256"), "raw response body_sha256")


def validate_transport_failure_record(record: Mapping[str, object]) -> None:
    _require_exact_mapping_keys(
        record,
        "transport failure",
        {
            "schema_version",
            "case_id",
            "request_bytes_sha256",
            "exception_class",
            "exception_message",
        },
    )
    if record.get("schema_version") != "semantic_anchor_transport_failure_v1":
        raise ValueError("transport failure schema_version mismatch")
    _validate_case_id(record.get("case_id"), "transport failure case_id")
    _validate_sha256_string(record.get("request_bytes_sha256"), "request_bytes_sha256")
    _require_nonempty_string(record.get("exception_class"), "exception_class")
    if not isinstance(record.get("exception_message"), str):
        raise ValueError("exception_message must be a string")


def validate_case_receipt(record: Mapping[str, object]) -> None:
    required = {"schema_version", "case_id", "request_sha256", "machine_status"}
    allowed = required.union({"raw_response_sha256"})
    keys = set(record)
    if not required.issubset(keys) or keys.difference(allowed):
        raise ValueError(
            f"case receipt keys mismatch: missing={required - keys} extra={keys - allowed}"
        )
    if record.get("schema_version") != "semantic_anchor_case_receipt_v1":
        raise ValueError("case receipt schema_version mismatch")
    _validate_case_id(record.get("case_id"), "case receipt case_id")
    _validate_sha256_string(record.get("request_sha256"), "case receipt request_sha256")
    status = record.get("machine_status")
    if status not in {"mechanical_pass", "mechanical_no_go"}:
        raise ValueError("case receipt machine_status mismatch")
    if "raw_response_sha256" in record:
        _validate_sha256_string(
            record.get("raw_response_sha256"), "case receipt raw_response_sha256"
        )
    elif status == "mechanical_pass":
        raise ValueError("mechanical_pass requires raw_response_sha256")


def validate_gold_label_case(
    gold_case: Mapping[str, object], *, case_id: str, u1_token_count: int = 5
) -> None:
    _require_exact_mapping_keys(
        gold_case,
        "gold label",
        {
            "schema_version",
            "case_id",
            "decision",
            "acceptable_ranges",
            "wrong_referent_ranges",
        }
        if gold_case.get("decision") == "select"
        else {
            "schema_version",
            "case_id",
            "decision",
            "acceptable_ranges",
            "wrong_referent_ranges",
            "abstain_reason",
        },
    )
    if gold_case.get("schema_version") != "semantic_anchor_gold_label_v1":
        raise ValueError("gold label schema_version mismatch")
    if gold_case.get("case_id") != case_id:
        raise ValueError("gold label case_id mismatch")
    decision = gold_case.get("decision")
    acceptable_ranges = _range_tuple(
        gold_case.get("acceptable_ranges"),
        "acceptable_ranges",
        u1_token_count=u1_token_count,
    )
    wrong_ranges = _range_tuple(
        gold_case.get("wrong_referent_ranges"),
        "wrong_referent_ranges",
        u1_token_count=u1_token_count,
    )
    if set(acceptable_ranges).intersection(wrong_ranges):
        raise ValueError("gold acceptable and wrong ranges overlap")
    if decision == "select":
        if not acceptable_ranges:
            raise ValueError("select gold label requires acceptable ranges")
        if "abstain_reason" in gold_case:
            raise ValueError("select gold label must not include abstain_reason")
    elif decision == "abstain":
        if acceptable_ranges:
            raise ValueError("abstain gold label must not include acceptable ranges")
        _required_enum(gold_case, "abstain_reason", ABSTAIN_REASONS)
    else:
        raise ValueError("gold label decision must be select or abstain")


def validate_gold_label_bundle(
    gold: Mapping[str, object],
    *,
    case_order: Sequence[str],
    u1_token_count: int = 5,
) -> None:
    if gold.get("schema_version") != "semantic_anchor_gold_labels_v1":
        raise ValueError("gold labels schema_version mismatch")
    cases = gold.get("cases")
    if not isinstance(cases, list):
        raise ValueError("gold labels cases must be list")
    if [case.get("case_id") for case in cases if isinstance(case, dict)] != list(case_order):
        raise ValueError("gold labels case order mismatch")
    for case_id, raw_case in zip(case_order, cases, strict=True):
        validate_gold_label_case(
            _as_mapping(raw_case, "gold label"),
            case_id=case_id,
            u1_token_count=u1_token_count,
        )


def classify_model_response(
    response: Mapping[str, object],
    *,
    gold_case: Mapping[str, object],
    case_id: str,
    u1_token_count: int = 5,
) -> str:
    """Classify one synthetic response against scorer-only gold."""

    validate_gold_label_case(
        gold_case,
        case_id=case_id,
        u1_token_count=u1_token_count,
    )
    try:
        validate_model_response_shape(
            response,
            case_id=case_id,
            u1_token_count=u1_token_count,
        )
    except ValueError:
        return "mechanical_failure"

    gold_decision = gold_case["decision"]
    response_decision = response["decision"]
    if response_decision == "abstain":
        return "safe_abstain" if gold_decision == "abstain" else "wrong_abstain"

    response_range = (int(response["start_token"]), int(response["end_token"]))
    acceptable_ranges = _range_tuple(
        gold_case.get("acceptable_ranges"),
        "acceptable_ranges",
        u1_token_count=u1_token_count,
    )
    if response_range in acceptable_ranges:
        return "correct_select"
    wrong_ranges = _range_tuple(
        gold_case.get("wrong_referent_ranges"),
        "wrong_referent_ranges",
        u1_token_count=u1_token_count,
    )
    if response_range in wrong_ranges or gold_decision == "abstain":
        return "wrong_referent"
    return "wrong_referent"


def build_scorer_receipt(
    responses_by_case_id: Mapping[str, Mapping[str, object]],
    *,
    gold: Mapping[str, object],
    case_order: Sequence[str],
    u1_token_count: int = 5,
) -> dict[str, object]:
    """Build a deterministic scorer receipt from sealed synthetic responses."""

    validate_gold_label_bundle(gold, case_order=case_order, u1_token_count=u1_token_count)
    gold_by_case = {
        str(case["case_id"]): _as_mapping(case, "gold label")
        for case in _as_mapping(gold, "gold labels")["cases"]  # type: ignore[index]
        if isinstance(case, dict)
    }
    case_results = []
    for case_id in case_order:
        response = responses_by_case_id.get(case_id)
        classification = (
            "mechanical_failure"
            if response is None
            else classify_model_response(
                response,
                gold_case=gold_by_case[case_id],
                case_id=case_id,
                u1_token_count=u1_token_count,
            )
        )
        case_results.append({"case_id": case_id, "classification": classification})
    receipt = {
        "schema_version": "semantic_anchor_scorer_receipt_v1",
        "case_results": case_results,
    }
    validate_scorer_receipt(receipt, case_order=case_order)
    return receipt


def validate_scorer_receipt(
    receipt: Mapping[str, object], *, case_order: Sequence[str]
) -> None:
    _require_exact_mapping_keys(receipt, "scorer receipt", {"schema_version", "case_results"})
    if receipt.get("schema_version") != "semantic_anchor_scorer_receipt_v1":
        raise ValueError("scorer receipt schema_version mismatch")
    case_results = receipt.get("case_results")
    if not isinstance(case_results, list):
        raise ValueError("scorer receipt case_results must be list")
    if len(case_results) != len(case_order):
        raise ValueError("scorer receipt case count mismatch")
    observed_case_ids: list[str] = []
    for result in case_results:
        row = _as_mapping(result, "scorer result")
        _require_exact_mapping_keys(row, "scorer result", {"case_id", "classification"})
        case_id = _validate_case_id(row.get("case_id"), "scorer case_id")
        classification = row.get("classification")
        if classification not in SCORER_CLASSIFICATIONS:
            raise ValueError("scorer classification mismatch")
        observed_case_ids.append(case_id)
    if tuple(observed_case_ids) != tuple(case_order):
        raise ValueError("scorer receipt case order mismatch")


def validate_reviewer_receipt(receipt: Mapping[str, object]) -> None:
    _require_exact_mapping_keys(
        receipt,
        "reviewer receipt",
        {
            "schema_version",
            "reviewer_count",
            "unanimous",
            "scorer_review_sha256",
            "sealed_responses_sha256",
            "gold_sha256",
            "rubric_sha256",
            "artifact_bundle_sha256",
        },
    )
    if receipt.get("schema_version") != "semantic_anchor_reviewer_receipt_v1":
        raise ValueError("reviewer receipt schema_version mismatch")
    reviewer_count = _strict_int(receipt.get("reviewer_count"), "reviewer_count")
    if reviewer_count < 2:
        raise ValueError("reviewer receipt requires at least two reviewers")
    if receipt.get("unanimous") is not True:
        raise ValueError("reviewer receipt must be unanimous")
    for key in (
        "scorer_review_sha256",
        "sealed_responses_sha256",
        "gold_sha256",
        "rubric_sha256",
        "artifact_bundle_sha256",
    ):
        _validate_sha256_string(receipt.get(key), f"reviewer receipt {key}")


def synthetic_qualification_terminal_state(
    *,
    scorer_receipt: Mapping[str, object],
    reviewer_receipt: Mapping[str, object],
    case_order: Sequence[str],
) -> str:
    """Map scorer/reviewer receipts to the final synthetic terminal state."""

    validate_scorer_receipt(scorer_receipt, case_order=case_order)
    validate_reviewer_receipt(reviewer_receipt)
    classifications = [
        _as_mapping(row, "scorer result")["classification"]
        for row in _as_mapping(scorer_receipt, "scorer receipt")["case_results"]  # type: ignore[index]
    ]
    if all(
        classification in {"correct_select", "safe_abstain"}
        for classification in classifications
    ):
        return TerminalState.COMPLETED_SYNTHETIC_GO.value
    return TerminalState.COMPLETED_QUALIFICATION_NO_GO.value


def validate_ledger_prefix(
    *,
    case_order: Sequence[str],
    reservations: Sequence[Mapping[str, object]],
    dispatches: Sequence[Mapping[str, object]],
    raw_responses: Sequence[Mapping[str, object]],
    transport_failures: Sequence[Mapping[str, object]],
    case_receipts: Sequence[Mapping[str, object]],
    terminal_receipt: Mapping[str, object],
    run_manifest: Mapping[str, object] | None = None,
) -> None:
    """Validate a synthetic v4 ledger prefix without opening gold or model files."""

    validate_fixed_case_order(case_order)

    for record in reservations:
        validate_reservation(record)
    for record in dispatches:
        validate_dispatch_record(record)
    for record in raw_responses:
        validate_raw_response_record(record)
    for record in transport_failures:
        validate_transport_failure_record(record)
    for record in case_receipts:
        validate_case_receipt(record)
    validate_terminal_receipt(terminal_receipt)
    if run_manifest is not None:
        validate_run_manifest(run_manifest)
        if run_manifest.get("artifact_sha256") != terminal_receipt.get("artifact_sha256"):
            raise ValueError("run manifest and terminal receipt artifact hashes differ")

    _require_unique_case_records(reservations, "reservation")
    _require_unique_case_records(dispatches, "dispatch")
    _require_unique_case_records(raw_responses, "raw response")
    _require_unique_case_records(transport_failures, "transport failure")
    _require_unique_case_records(case_receipts, "case receipt")

    order_index = {case_id: index for index, case_id in enumerate(case_order)}
    observed_case_ids = [
        str(record["case_id"])
        for records in (
            reservations,
            dispatches,
            raw_responses,
            transport_failures,
            case_receipts,
        )
        for record in records
    ]
    if any(case_id not in order_index for case_id in observed_case_ids):
        raise ValueError("ledger prefix contains unknown case_id")
    if observed_case_ids:
        max_index = max(order_index[case_id] for case_id in observed_case_ids)
        expected_prefix = set(case_order[: max_index + 1])
        if not set(observed_case_ids).issubset(expected_prefix):
            raise ValueError("ledger prefix contains non-prefix case_id")
        for earlier_case_id in case_order[:max_index]:
            if earlier_case_id not in {str(row["case_id"]) for row in reservations}:
                raise ValueError("ledger prefix skips an earlier reservation")

    reservation_by_case = {str(record["case_id"]): record for record in reservations}
    dispatch_by_case = {str(record["case_id"]): record for record in dispatches}
    raw_by_case = {str(record["case_id"]): record for record in raw_responses}
    failure_by_case = {str(record["case_id"]): record for record in transport_failures}
    receipt_by_case = {str(record["case_id"]): record for record in case_receipts}
    if set(raw_by_case).intersection(failure_by_case):
        raise ValueError("case cannot have both raw response and transport failure")

    for case_id, dispatch in dispatch_by_case.items():
        reservation = reservation_by_case.get(case_id)
        if reservation is None:
            raise ValueError("dispatch lacks reservation")
        if dispatch["request_bytes_sha256"] != reservation["request_sha256"]:
            raise ValueError("dispatch request hash differs from reservation")
    for case_id, raw in raw_by_case.items():
        if case_id not in dispatch_by_case:
            raise ValueError("raw response lacks dispatch")
        receipt = receipt_by_case.get(case_id)
        if receipt is None:
            raise ValueError("raw response lacks case receipt")
        if receipt.get("machine_status") != "mechanical_pass":
            raise ValueError("raw response case receipt must be mechanical_pass")
        if receipt.get("raw_response_sha256") != raw["body_sha256"]:
            raise ValueError("case receipt raw hash differs from raw response")
    for case_id, failure in failure_by_case.items():
        if case_id not in dispatch_by_case:
            raise ValueError("transport failure lacks dispatch")
        if failure["request_bytes_sha256"] != dispatch_by_case[case_id]["request_bytes_sha256"]:
            raise ValueError("transport failure request hash differs from dispatch")
        receipt = receipt_by_case.get(case_id)
        if receipt is not None and receipt.get("machine_status") != "mechanical_no_go":
            raise ValueError("transport failure receipt must be mechanical_no_go")
    for case_id, receipt in receipt_by_case.items():
        reservation = reservation_by_case.get(case_id)
        if reservation is None:
            raise ValueError("case receipt lacks reservation")
        if receipt["request_sha256"] != reservation["request_sha256"]:
            raise ValueError("case receipt request hash differs from reservation")

    attempted = _strict_int(terminal_receipt.get("attempted_calls"), "attempted_calls")
    completed = _strict_int(terminal_receipt.get("completed_calls"), "completed_calls")
    raw_committed = _strict_int(
        terminal_receipt.get("raw_committed_calls"), "raw_committed_calls"
    )
    if attempted != len(dispatches):
        raise ValueError("terminal attempted_calls differs from dispatch count")
    if completed != len(raw_responses):
        raise ValueError("terminal completed_calls differs from raw response count")
    if raw_committed != len(raw_responses):
        raise ValueError("terminal raw_committed_calls differs from raw response count")


def validate_model_inventory(inventory: Mapping[str, object]) -> None:
    _require_exact_mapping_keys(
        inventory,
        "model inventory",
        {
            "schema_version",
            "repository",
            "revision",
            "quantization_method",
            "safetensors_index_total_size",
            "snapshot_path",
            "loaded_shards",
            "loaded_files",
            "unloaded_files",
            "denied_files",
            "file_sha256",
        },
    )
    if inventory.get("schema_version") != "semantic_anchor_model_inventory_attestation_v1":
        raise ValueError("model inventory schema_version mismatch")
    if inventory.get("repository") != "openai/gpt-oss-20b":
        raise ValueError("unexpected model repository")
    expected_revision = "6cee5e81ee83917806bbde320786a8fb61efebee"
    if inventory.get("revision") != expected_revision:
        raise ValueError("unexpected model revision")
    if inventory.get("quantization_method") != "mxfp4":
        raise ValueError("unexpected quantization method")
    if inventory.get("safetensors_index_total_size") != 13_761_264_768:
        raise ValueError("unexpected safetensors index total size")
    snapshot_path = inventory.get("snapshot_path")
    if not isinstance(snapshot_path, str) or expected_revision not in snapshot_path:
        raise ValueError("model inventory snapshot_path must bind expected revision")
    loaded = _string_tuple(inventory.get("loaded_shards"), "loaded_shards")
    if len(loaded) != 3 or len(set(loaded)) != 3:
        raise ValueError("model inventory must have exactly three unique loaded shards")
    loaded_files = set(_string_tuple(inventory.get("loaded_files"), "loaded_files"))
    unloaded_files = set(_string_tuple(inventory.get("unloaded_files"), "unloaded_files"))
    denied = set(_string_tuple(inventory.get("denied_files"), "denied_files"))
    if "original/model.safetensors" not in denied:
        raise ValueError("original/model.safetensors must be denied or hidden from loader")
    if "original/model.safetensors" in loaded_files or "original/model.safetensors" in loaded:
        raise ValueError("original/model.safetensors must not be loaded")
    if "original/model.safetensors" not in unloaded_files:
        raise ValueError("original/model.safetensors must be listed as unloaded")
    overlap = loaded_files.intersection(denied)
    if overlap:
        raise ValueError(f"model inventory denied files cannot be loaded: {sorted(overlap)}")
    if not set(loaded).issubset(loaded_files):
        raise ValueError("model inventory loaded_shards must be loaded_files")
    file_sha256 = inventory.get("file_sha256")
    if not isinstance(file_sha256, Mapping) or not file_sha256:
        raise ValueError("model inventory file_sha256 must be a nonempty object")
    for path in loaded_files.union(unloaded_files).union(denied):
        digest = file_sha256.get(path)
        if not isinstance(digest, str) or len(digest) != 64 or digest.lower() != digest:
            raise ValueError(f"model inventory missing lowercase sha256 for {path}")
        if any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"model inventory invalid sha256 for {path}")


def validate_fixed_case_order(case_order: Sequence[str]) -> None:
    expected = tuple(f"synthetic-case-{index:03d}" for index in range(1, 25))
    if tuple(case_order) != expected:
        raise ValueError("case_order_mismatch")


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


def validate_replay_mutation_registry(registry: Mapping[str, object]) -> tuple[str, ...]:
    if registry.get("schema_version") != "semantic_anchor_replay_mutation_registry_v1":
        raise ValueError("replay mutation registry schema_version mismatch")
    mutations = registry.get("mutations")
    if not isinstance(mutations, list):
        raise ValueError("replay mutation registry mutations must be list")
    mutation_ids: list[str] = []
    failure_codes: set[str] = set()
    for mutation in mutations:
        entry = _as_mapping(mutation, "replay mutation")
        mutation_id = entry.get("mutation_id")
        failure_code = entry.get("expected_failure_code")
        target = entry.get("target_artifact")
        if not isinstance(mutation_id, str) or not mutation_id:
            raise ValueError("replay mutation_id must be nonempty string")
        if not isinstance(failure_code, str) or not failure_code:
            raise ValueError("replay expected_failure_code must be nonempty string")
        if not isinstance(target, str) or not target:
            raise ValueError("replay target_artifact must be nonempty string")
        mutation_ids.append(mutation_id)
        failure_codes.add(failure_code)
    if tuple(mutation_ids) != REPLAY_MUTATION_IDS:
        raise ValueError("replay mutation IDs drifted")
    if tuple(
        _as_mapping(mutation, "replay mutation").get("expected_failure_code")
        for mutation in mutations
    ) != REPLAY_FAILURE_CODES:
        raise ValueError("replay mutation failure codes drifted")
    if len(failure_codes) != len(mutation_ids):
        raise ValueError("replay mutations must have unique failure codes")
    return tuple(mutation_ids)


def run_replay_mutation_oracle(mutation_id: str) -> str:
    """Execute one topic-free replay mutation oracle and return its failure code."""

    if mutation_id not in REPLAY_MUTATION_IDS:
        raise ValueError(f"unknown replay mutation: {mutation_id}")
    failure_code = REPLAY_FAILURE_CODES[REPLAY_MUTATION_IDS.index(mutation_id)]
    try:
        _execute_replay_mutation_oracle(mutation_id)
    except ValueError:
        return failure_code
    raise ValueError(f"replay mutation did not fail: {mutation_id}")


def _execute_replay_mutation_oracle(mutation_id: str) -> None:
    case_order = tuple(f"synthetic-case-{index:03d}" for index in range(1, 25))
    artifact_sha256 = {"manifest": "a" * 64}
    if mutation_id == "mutate_request_body_byte":
        validate_ledger_prefix(
            case_order=case_order,
            reservations=[_oracle_reservation(request_sha256="1" * 64)],
            dispatches=[_oracle_dispatch(request_sha256="9" * 64)],
            raw_responses=[],
            transport_failures=[],
            case_receipts=[],
            terminal_receipt=_oracle_terminal(attempted=1, completed=0, raw_committed=0),
        )
    elif mutation_id == "mutate_raw_response_body_byte":
        validate_ledger_prefix(
            case_order=case_order,
            reservations=[_oracle_reservation()],
            dispatches=[_oracle_dispatch()],
            raw_responses=[_oracle_raw_response(body_sha256="2" * 64)],
            transport_failures=[],
            case_receipts=[_oracle_case_receipt(raw_response_sha256="9" * 64)],
            terminal_receipt=_oracle_terminal(attempted=1, completed=1, raw_committed=1),
        )
    elif mutation_id == "swap_case_order_ids":
        swapped = list(case_order)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        validate_fixed_case_order(swapped)
    elif mutation_id == "mark_prefix_gold_opened":
        validate_terminal_receipt(
            _oracle_terminal(
                attempted=1,
                completed=0,
                raw_committed=0,
                gold_opened=True,
            )
        )
    elif mutation_id == "change_response_schema_case_const":
        schema = expected_case001_response_schema()
        schema = dict(schema)
        properties = dict(_as_mapping(schema["properties"], "schema properties"))
        properties["case_id"] = {"const": "synthetic-case-999", "type": "string"}
        schema["properties"] = properties
        if schema != expected_case001_response_schema():
            raise ValueError("schema_constant_mismatch")
    elif mutation_id == "add_gold_to_runner_manifest":
        validate_runner_gold_separation(
            ("semantic_anchor_gold_labels_v1.json",),
            scorer_only_artifacts(),
        )
    elif mutation_id == "change_model_inventory_hash":
        attestation = _oracle_pre_dispatch_attestation(model_inventory_sha256="9" * 64)
        validate_pre_dispatch_attestation(attestation)
        if attestation["model_inventory_sha256"] != "8" * 64:
            raise ValueError("model_inventory_changed")
    elif mutation_id == "mark_dispatch_non_loopback":
        validate_dispatch_record(_oracle_dispatch(loopback_only=False))
    else:
        raise ValueError(f"unknown replay mutation: {mutation_id}")


def _oracle_reservation(
    *, case_id: str = "synthetic-case-001", request_sha256: str = "1" * 64
) -> dict[str, object]:
    return {
        "schema_version": "semantic_anchor_reservation_v1",
        "run_id": "oracle-run",
        "case_id": case_id,
        "request_sha256": request_sha256,
        "create_only": True,
    }


def _oracle_dispatch(
    *,
    case_id: str = "synthetic-case-001",
    request_sha256: str = "1" * 64,
    loopback_only: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": "semantic_anchor_dispatch_record_v1",
        "case_id": case_id,
        "request_bytes_sha256": request_sha256,
        "loopback_only": loopback_only,
        "dispatch_counted": True,
    }


def _oracle_raw_response(
    *, case_id: str = "synthetic-case-001", body_sha256: str = "2" * 64
) -> dict[str, object]:
    return {
        "schema_version": "semantic_anchor_raw_response_body_v1",
        "case_id": case_id,
        "http_status": 200,
        "finish_reason": "stop",
        "served_model": "gpt-oss-local",
        "body_size_bytes": 1,
        "body_sha256": body_sha256,
    }


def _oracle_case_receipt(
    *,
    case_id: str = "synthetic-case-001",
    request_sha256: str = "1" * 64,
    raw_response_sha256: str = "2" * 64,
) -> dict[str, object]:
    return {
        "schema_version": "semantic_anchor_case_receipt_v1",
        "case_id": case_id,
        "request_sha256": request_sha256,
        "machine_status": "mechanical_pass",
        "raw_response_sha256": raw_response_sha256,
    }


def _oracle_terminal(
    *,
    attempted: int,
    completed: int,
    raw_committed: int,
    gold_opened: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "semantic_anchor_terminal_receipt_v1",
        "terminal_state": "interrupted_incomplete",
        "attempted_calls": attempted,
        "completed_calls": completed,
        "raw_committed_calls": raw_committed,
        "gold_opened": gold_opened,
        "artifact_sha256": {"manifest": "a" * 64},
    }


def _oracle_pre_dispatch_attestation(
    *, model_inventory_sha256: str = "8" * 64
) -> dict[str, object]:
    return {
        "schema_version": "semantic_anchor_pre_dispatch_attestation_v1",
        "attestation_id": "oracle-attestation",
        "served_model": "gpt-oss-local",
        "egress_denied": True,
        "read_only_model_mount": True,
        "model_inventory_sha256": model_inventory_sha256,
    }


def validate_renderer_oracle(oracle: Mapping[str, object]) -> tuple[str, ...]:
    if oracle.get("schema_version") != "semantic_anchor_renderer_oracle_v1":
        raise ValueError("renderer oracle schema_version mismatch")
    cases = oracle.get("oracle_cases")
    if not isinstance(cases, list):
        raise ValueError("renderer oracle cases must be list")
    oracle_ids: list[str] = []
    for value in cases:
        case = _as_mapping(value, "renderer oracle case")
        oracle_id = case.get("oracle_id")
        if not isinstance(oracle_id, str) or not oracle_id:
            raise ValueError("renderer oracle_id must be nonempty string")
        oracle_ids.append(oracle_id)
        input_value = _as_mapping(case.get("input"), "renderer oracle input")
        case_id = input_value.get("case_id")
        if not isinstance(case_id, str) or not case_id.startswith("synthetic-case-"):
            raise ValueError("renderer oracle case_id must be synthetic")
        anchor_range = _as_mapping(input_value.get("anchor_range"), "anchor range")
        validate_model_response_shape(
            {
                "schema_version": MODEL_RESPONSE_SCHEMA_VERSION,
                "case_id": case_id,
                "decision": "abstain" if anchor_range.get("start_token") == -1 else "select",
                "start_token": anchor_range.get("start_token"),
                "end_token": anchor_range.get("end_token"),
            },
            case_id=case_id,
            u1_token_count=7,
        )
        expected = _as_mapping(case.get("expected"), "renderer oracle expected")
        status = expected.get("status")
        if status == "rendered":
            facets = expected.get("facet_queries")
            if not isinstance(facets, list) or len(facets) != 2:
                raise ValueError("rendered oracle must have two facet queries")
            if any(audit_denied_path_fragments(str(facet)) for facet in facets):
                raise ValueError("renderer oracle facet contains denied path fragment")
        elif status == "original_only":
            if expected.get("facet_queries") != []:
                raise ValueError("original_only oracle must have no facet queries")
        else:
            raise ValueError("renderer oracle status must be rendered or original_only")
    if tuple(oracle_ids) != RENDERER_ORACLE_IDS:
        raise ValueError("renderer oracle IDs drifted")
    return tuple(oracle_ids)


def validate_import_open_audit_fixture(audit: Mapping[str, object]) -> None:
    if audit.get("schema_version") != "semantic_anchor_import_open_audit_v1":
        raise ValueError("import/open audit schema_version mismatch")
    if tuple(audit.get("allowed_import_roots", ())) != ALLOWED_IMPORT_ROOTS:
        raise ValueError("allowed import roots drifted")
    if tuple(audit.get("denied_imports", ())) != tuple(sorted(DENIED_IMPORTS)):
        raise ValueError("denied imports drifted")
    if tuple(audit.get("denied_path_fragments", ())) != DENIED_PATH_FRAGMENTS:
        raise ValueError("denied path fragments drifted")
    if audit.get("allowed_artifact_directory") != "docs/superpowers/det_sparse_v4_contract_artifacts":
        raise ValueError("allowed artifact directory drifted")


def validate_artifact_bundle(
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, str]:
    """Validate the complete offline artifact bundle, including scorer-only gold."""

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
    _validate_artifact_cross_file_consistency(artifact_dir, include_scorer_only=True)
    return hashes


def validate_runner_artifact_bundle(
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, str]:
    """Validate and hash only artifacts visible to the runner.

    This deliberately does not open scorer-only artifacts such as gold labels.
    The scorer validates those artifacts after the raw response ledger is sealed.
    """

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
            path = artifact_dir / path_value
            if not path.is_file():
                raise ValueError(f"artifact missing: {path_value}")
            hashes[path_value] = sha256_file(path)
        elif visibility == "scorer":
            scorer_files.append(path_value)
        elif visibility != "reviewer":
            raise ValueError(f"unknown artifact visibility for {path_value}: {visibility}")

    missing_runner = set(runner_visible_artifacts()).difference(runner_files)
    if missing_runner:
        raise ValueError(f"manifest missing runner artifacts: {sorted(missing_runner)}")
    missing_scorer = set(scorer_only_artifacts()).difference(scorer_files)
    if missing_scorer:
        raise ValueError(f"manifest missing scorer artifacts: {sorted(missing_scorer)}")
    validate_runner_gold_separation(runner_files, scorer_files)
    _validate_artifact_cross_file_consistency(artifact_dir, include_scorer_only=False)
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
        "semantic_anchor_reservation_v1.schema.json",
        "semantic_anchor_pre_dispatch_attestation_v1.schema.json",
        "semantic_anchor_dispatch_record_v1.schema.json",
        "semantic_anchor_raw_response_body_v1.schema.json",
        "semantic_anchor_transport_failure_v1.schema.json",
        "semantic_anchor_case_receipt_v1.schema.json",
        "semantic_anchor_run_manifest_v1.schema.json",
        "semantic_anchor_replay_mutation_registry_v1.json",
        "semantic_anchor_renderer_oracle_v1.json",
        "semantic_anchor_model_inventory_attestation_v1.json",
        "semantic_anchor_import_open_audit_v1.json",
    )


def scorer_only_artifacts() -> tuple[str, ...]:
    return (
        "semantic_anchor_gold_labels_v1.json",
        "semantic_anchor_scorer_v1.schema.json",
        "semantic_anchor_reviewer_rubric_v1.md",
        "semantic_anchor_reviewer_receipt_v1.schema.json",
        "semantic_anchor_reviewer_qualification_review_v1.schema.json",
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


def _validate_artifact_cross_file_consistency(
    artifact_dir: Path,
    *,
    include_scorer_only: bool,
) -> None:
    response_schema = load_json_no_duplicates(
        artifact_dir / "semantic_anchor_response_v1.case001.schema.json"
    )
    if response_schema != expected_case001_response_schema():
        raise ValueError("case001 response schema fixture does not match generator")
    _validate_offset_schema_fixtures(artifact_dir)

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
    replay_mutations = _as_mapping(
        load_json_no_duplicates(
            artifact_dir / "semantic_anchor_replay_mutation_registry_v1.json"
        ),
        "replay mutation registry",
    )
    validate_replay_mutation_registry(replay_mutations)
    _validate_ledger_schema_fixtures(artifact_dir)
    renderer_oracle = _as_mapping(
        load_json_no_duplicates(artifact_dir / "semantic_anchor_renderer_oracle_v1.json"),
        "renderer oracle",
    )
    validate_renderer_oracle(renderer_oracle)
    model_inventory = _as_mapping(
        load_json_no_duplicates(
            artifact_dir / "semantic_anchor_model_inventory_attestation_v1.json"
        ),
        "model inventory attestation",
    )
    validate_model_inventory(model_inventory)
    import_open_audit = _as_mapping(
        load_json_no_duplicates(artifact_dir / "semantic_anchor_import_open_audit_v1.json"),
        "import open audit",
    )
    validate_import_open_audit_fixture(import_open_audit)

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

    gold_by_id: dict[object, Mapping[str, object]] = {}
    if include_scorer_only:
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
        gold_case = (
            _as_mapping(gold_by_id[case_id], "gold case")
            if include_scorer_only
            else None
        )
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
    gold_case: Mapping[str, object] | None,
    request: Mapping[str, object],
    registry_case: Mapping[str, object],
    system_prompt: str,
) -> None:
    if case.get("schema_version") != "semantic_anchor_synthetic_case_v1":
        raise ValueError(f"case schema_version mismatch: {case_id}")
    if case.get("case_id") != case_id:
        raise ValueError(f"case ID mismatch: {case_id}")
    if gold_case is not None:
        if gold_case.get("schema_version") != "semantic_anchor_gold_label_v1":
            raise ValueError(f"gold schema_version mismatch: {case_id}")
        if gold_case.get("case_id") != case_id:
            raise ValueError(f"gold ID mismatch: {case_id}")
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
    ranges: list[object] = []
    if gold_case is not None:
        if gold_case.get("decision") != decision:
            raise ValueError(f"gold decision differs from registry: {case_id}")
        validate_gold_label_case(gold_case, case_id=case_id, u1_token_count=5)
        raw_ranges = gold_case.get("acceptable_ranges")
        if not isinstance(raw_ranges, list):
            raise ValueError(f"acceptable_ranges must be list: {case_id}")
        ranges = raw_ranges
    if decision == "select":
        if gold_case is not None:
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
        if gold_case is not None:
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


def _validate_ledger_schema_fixtures(artifact_dir: Path) -> None:
    for filename in LEDGER_SCHEMA_FILES:
        schema = _as_mapping(load_json_no_duplicates(artifact_dir / filename), filename)
        properties = _as_mapping(schema.get("properties"), f"{filename} properties")
        if schema.get("type") != "object":
            raise ValueError(f"ledger schema must be object: {filename}")
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"ledger schema must reject extra properties: {filename}")
        schema_version = _as_mapping(
            properties.get("schema_version"), f"{filename} schema_version"
        )
        expected_const = filename.removesuffix(".schema.json")
        if schema_version.get("const") != expected_const:
            raise ValueError(f"ledger schema_version const mismatch: {filename}")
        required = schema.get("required")
        if not isinstance(required, list) or "schema_version" not in required:
            raise ValueError(f"ledger schema must require schema_version: {filename}")


def _validate_offset_schema_fixtures(artifact_dir: Path) -> None:
    health_schema = _as_mapping(
        load_json_no_duplicates(artifact_dir / "lucene_whole_unit_offsets_health_v1.schema.json"),
        "offset health schema",
    )
    error_schema = _as_mapping(
        load_json_no_duplicates(artifact_dir / "lucene_whole_unit_offsets_error_v1.schema.json"),
        "offset error schema",
    )
    response_schema = _as_mapping(
        load_json_no_duplicates(artifact_dir / "lucene_whole_unit_offsets_response_v1.schema.json"),
        "offset response schema",
    )
    request_schema = _as_mapping(
        load_json_no_duplicates(artifact_dir / "lucene_whole_unit_offsets_request_v1.schema.json"),
        "offset request schema",
    )
    for name, schema, version in (
        ("offset health schema", health_schema, "lucene_whole_unit_offsets_health_v1"),
        ("offset error schema", error_schema, "lucene_whole_unit_offsets_error_v1"),
        ("offset response schema", response_schema, OFFSET_RESPONSE_SCHEMA_VERSION),
        ("offset request schema", request_schema, OFFSET_REQUEST_SCHEMA_VERSION),
    ):
        if schema.get("type") != "object":
            raise ValueError(f"{name} must be object")
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"{name} must reject extra properties")
        properties = _as_mapping(schema.get("properties"), f"{name} properties")
        schema_version = _as_mapping(
            properties.get("schema_version"), f"{name} schema_version"
        )
        if schema_version.get("const") != version:
            raise ValueError(f"{name} schema_version const mismatch")
        required = schema.get("required")
        if not isinstance(required, list) or "schema_version" not in required:
            raise ValueError(f"{name} must require schema_version")
    health_properties = _as_mapping(health_schema.get("properties"), "health properties")
    if _as_mapping(health_properties.get("legacy_analyzer_port"), "legacy port").get("const") != 18081:
        raise ValueError("offset health legacy port const mismatch")
    if _as_mapping(health_properties.get("offset_analyzer_port"), "offset port").get("const") != 18082:
        raise ValueError("offset health offset port const mismatch")
    error_code = _as_mapping(
        _as_mapping(error_schema.get("properties"), "error properties").get("error_code"),
        "error_code schema",
    )
    if tuple(error_code.get("enum", ())) != (
        "bad_json",
        "schema_mismatch",
        "invalid_utf8",
        "analyzer_failure",
    ):
        raise ValueError("offset error_code enum drifted")


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_mapping_keys(
    value: Mapping[str, object], name: str, expected_keys: set[str]
) -> None:
    keys = set(value)
    if keys != expected_keys:
        raise ValueError(
            f"{name} keys mismatch: missing={expected_keys - keys} extra={keys - expected_keys}"
        )


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _loads_object_no_duplicates(text: str, name: str) -> Mapping[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_case_id(value: object, name: str) -> str:
    candidate = _require_nonempty_string(value, name)
    prefix = "synthetic-case-"
    suffix = candidate.removeprefix(prefix)
    if (
        not candidate.startswith(prefix)
        or len(suffix) != 3
        or not suffix.isdigit()
        or not (1 <= int(suffix) <= 999)
    ):
        raise ValueError(f"{name} must be synthetic-case-NNN")
    return candidate


def _validate_sha256_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not _is_hex_sha256(value):
        raise ValueError(f"{name} must be lowercase sha256")
    return value


def _validate_sha256_mapping(value: object, name: str) -> None:
    mapping = _as_mapping(value, name)
    if not mapping:
        raise ValueError(f"{name} must be nonempty")
    for key, digest in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be nonempty strings")
        _validate_sha256_string(digest, f"{name}[{key}]")


def _range_tuple(
    value: object, name: str, *, u1_token_count: int
) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be list")
    ranges: list[tuple[int, int]] = []
    for raw_range in value:
        span = _as_mapping(raw_range, name)
        _require_exact_mapping_keys(span, name, {"start_token", "end_token"})
        start = _strict_int(span.get("start_token"), f"{name}.start_token")
        end = _strict_int(span.get("end_token"), f"{name}.end_token")
        if not (0 <= start < end <= u1_token_count):
            raise ValueError(f"{name} range must be nonempty and inside U1")
        ranges.append((start, end))
    if len(ranges) != len(set(ranges)):
        raise ValueError(f"{name} contains duplicate ranges")
    return tuple(ranges)


def _require_unique_case_records(
    records: Sequence[Mapping[str, object]], name: str
) -> None:
    seen: set[str] = set()
    for record in records:
        case_id = str(record["case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate {name} for {case_id}")
        seen.add(case_id)


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


def _is_hex_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


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
