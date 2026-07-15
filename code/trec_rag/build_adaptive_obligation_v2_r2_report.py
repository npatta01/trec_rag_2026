"""Build the portable R1-incident and R2-readiness technical report.

The builder authenticates already-frozen artifacts only.  It refuses any R2
approval, ledger, or proposal inventory and never constructs a model, runs
inference, performs retrieval, opens qrels, or authors a bespoke HTML runtime.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .adaptive_evidence_rank import verify_baseline_rankings
from .adaptive_obligation_v2_contract import verify_v2_contract
from .adaptive_obligation_v2_proposal_incident import verify_r1_incident
from .adaptive_obligation_v2_propose_r2 import verify_r2_proposal_preflight


TITLE = "Adaptive obligation proposal R2 readiness"
PILOT_TOPIC_IDS = ("219", "72", "300", "84")
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
R1_PROMPT_TOKENS = 54_402
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_BUILDER_PATH = "code/trec_rag/build_adaptive_obligation_v2_r2_report.py"
CANONICAL_RENDERER_PATH = Path(
    "/home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/"
    "0.2.8-13ceeea1f599/skills/build-report/scripts/"
    "deliver_portable_artifact.mjs"
)
CANONICAL_RENDERER_SHA256 = (
    "09d7afe25f76429ba1e99d08a5793af44c27738482d91e9b0bd2351ffd516066"
)
SOURCE_NAMES = ("contract", "baseline_rankings", "r1_incident", "r2_preflight")
SOURCE_LABELS = {
    "contract": "Verified adaptive obligation v2 contract receipt",
    "baseline_rankings": "Sealed NARRATIVE and FIXED-O0 ranking receipt",
    "r1_incident": "Verified immutable proposal R1 incident receipt",
    "r2_preflight": "Verified inference-free proposal R2 preflight receipt",
}
SUPPORTING_SOURCE_PATHS = {
    "recovery_plan": (
        "docs/superpowers/plans/"
        "2026-07-15-adaptive-obligation-proposal-r2-recovery.md"
    ),
    "recovery_design": (
        "docs/superpowers/specs/"
        "2026-07-15-adaptive-obligation-proposal-r2-recovery-design.md"
    ),
    "incident_implementation": (
        "code/trec_rag/adaptive_obligation_v2_proposal_incident.py"
    ),
    "r2_implementation": "code/trec_rag/adaptive_obligation_v2_propose_r2.py",
    "r2_model_implementation": (
        "code/trec_rag/adaptive_obligation_v2_local_model_r2.py"
    ),
    "report_builder": REPORT_BUILDER_PATH,
}
R2_FORBIDDEN_LEAVES = (
    "proposal_approval_r2.json",
    "proposal_ledger_r2",
    "proposals_r2",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
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


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _exact_integer(value: object, expected: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{label} differs")
    return value


def _exact_zero(value: object, label: str) -> None:
    _exact_integer(value, 0, label)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_repo_path(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or ".." in Path(value).parts
    ):
        raise ValueError(f"{label} must be a repository-relative path")
    return value


def _repo_relative(path: Path) -> str:
    resolved = Path(path).resolve(strict=True)
    try:
        return resolved.relative_to(REPO_ROOT.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ValueError(f"report source is outside the repository: {path}") from exc


def _present(path: Path) -> bool:
    return os.path.lexists(path)


def _utc_build_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validated_build_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("report build timestamp must be a UTC instant ending in Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("report build timestamp is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("report build timestamp must be UTC")
    return value


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _reject_r2_execution_artifacts(root: Path) -> None:
    if any(_present(root / name) for name in R2_FORBIDDEN_LEAVES):
        raise ValueError(
            "R2 inference must remain unopened: approval, ledger, and proposals "
            "must all be absent"
        )


def _incident_bound_path(
    incident: Mapping[str, object], key: str, *, directory: bool = False
) -> Path:
    binding = _object(incident.get(key), f"R1 incident {key} binding")
    relative = _safe_repo_path(binding.get("path"), f"R1 incident {key} path")
    repository = REPO_ROOT.resolve(strict=True)
    candidate = REPO_ROOT / relative
    try:
        candidate.resolve(strict=True).relative_to(repository)
    except (OSError, ValueError) as exc:
        raise ValueError(f"R1 incident {key} path is outside the repository") from exc
    if directory and not candidate.is_dir():
        raise ValueError(f"R1 incident {key} path is not a directory")
    if not directory and not candidate.is_file():
        raise ValueError(f"R1 incident {key} path is not a file")
    # The incident receipt hashes the path spelling as well as the bytes.  Replay
    # with the authenticated repository-relative spelling while the verifier
    # independently performs descriptor-safe reads.
    return Path(relative)


def load_verified_sources(
    *,
    contract_dir: Path,
    baseline_rankings_dir: Path,
    r1_incident_dir: Path,
    r2_preflight_dir: Path,
) -> dict[str, object]:
    """Authenticate the four report inputs without opening R2 execution."""

    contract_dir = Path(contract_dir)
    baseline_rankings_dir = Path(baseline_rankings_dir)
    r1_incident_dir = Path(r1_incident_dir)
    r2_preflight_dir = Path(r2_preflight_dir)
    if contract_dir.parent != r2_preflight_dir.parent:
        raise ValueError("v2 contract and R2 preflight roots differ")
    _reject_r2_execution_artifacts(r2_preflight_dir.parent)

    published_incident = _read_json_object(
        r1_incident_dir / "receipt.json", "R1 incident receipt"
    )
    r1_preflight_receipt = _incident_bound_path(published_incident, "preflight")
    r1_approval_path = _incident_bound_path(published_incident, "approval")
    r1_ledger_dir = _incident_bound_path(
        published_incident, "ledger", directory=True
    )

    receipts = {
        "contract": verify_v2_contract(contract_dir),
        "baseline_rankings": verify_baseline_rankings(baseline_rankings_dir),
        "r1_incident": verify_r1_incident(
            preflight_dir=r1_preflight_receipt.parent,
            approval_path=r1_approval_path,
            ledger_dir=r1_ledger_dir,
            output_dir=r1_incident_dir,
        ),
        "r2_preflight": verify_r2_proposal_preflight(r2_preflight_dir),
    }
    receipt_paths = {
        "contract": contract_dir / "receipt.json",
        "baseline_rankings": baseline_rankings_dir / "receipt.json",
        "r1_incident": r1_incident_dir / "receipt.json",
        "r2_preflight": r2_preflight_dir / "receipt.json",
    }
    return {
        **receipts,
        "source_paths": {
            name: _repo_relative(path) for name, path in receipt_paths.items()
        },
        "source_hashes": {
            name: _sha256_file(path) for name, path in receipt_paths.items()
        },
        "supporting_paths": dict(SUPPORTING_SOURCE_PATHS),
        "supporting_hashes": {
            name: _sha256_file(REPO_ROOT / path)
            for name, path in SUPPORTING_SOURCE_PATHS.items()
        },
        "r2_approval_present": False,
        "r2_ledger_present": False,
        "r2_proposals_present": False,
    }


def _verify_safety_counters(receipt: Mapping[str, object], label: str) -> None:
    if receipt.get("qrels_opened") is not False:
        raise ValueError(f"{label} qrels state differs")
    for field in (
        "network_call_count",
        "retrieval_call_count",
        "hosted_inference_call_count",
        "paid_call_count",
    ):
        _exact_zero(receipt.get(field), f"{label} {field}")


def _verified_report_values(sources: Mapping[str, object]) -> dict[str, object]:
    for field in (
        "r2_approval_present",
        "r2_ledger_present",
        "r2_proposals_present",
    ):
        if sources.get(field) is not False:
            raise ValueError(
                "R2 inference must remain unopened: approval, ledger, and proposals "
                "must all be absent"
            )

    contract = _object(sources.get("contract"), "v2 contract")
    baselines = _object(sources.get("baseline_rankings"), "baseline rankings")
    r1 = _object(sources.get("r1_incident"), "R1 incident")
    r2 = _object(sources.get("r2_preflight"), "R2 preflight")
    source_paths = _object(sources.get("source_paths"), "report source paths")
    source_hashes = _object(sources.get("source_hashes"), "report source hashes")
    supporting_paths = _object(
        sources.get("supporting_paths"), "report supporting paths"
    )
    supporting_hashes = _object(
        sources.get("supporting_hashes"), "report supporting hashes"
    )

    if any(
        receipt.get("topic_ids") != list(PILOT_TOPIC_IDS)
        for receipt in (contract, baselines, r2)
    ):
        raise ValueError("report topic boundary differs")
    _verify_safety_counters(contract, "v2 contract")
    _verify_safety_counters(baselines, "baseline rankings")
    _verify_safety_counters(r1, "R1 incident")
    _verify_safety_counters(r2, "R2 preflight")
    for receipt, label in ((contract, "v2 contract"), (baselines, "baselines"), (r2, "R2 preflight")):
        if receipt.get("external_cost_usd") != 0.0:
            raise ValueError(f"{label} external cost differs")

    if (
        contract.get("schema_version") != "adaptive-obligation-v2-contract-v1"
        or contract.get("status") != "complete"
    ):
        raise ValueError("v2 contract identity differs")
    _exact_integer(contract.get("parent_count"), 24, "parent count")
    _exact_integer(contract.get("reservoir_count"), 48, "reservoir count")
    _exact_integer(contract.get("unit_count"), 8_247, "unit count")
    _exact_zero(contract.get("protected_topic_count"), "protected topic count")
    _exact_zero(contract.get("model_load_count"), "contract model load count")
    _exact_zero(contract.get("inference_count"), "contract inference count")

    if (
        baselines.get("schema_version")
        != "adaptive-evidence-baseline-rankings-v1"
        or baselines.get("status") != "complete"
    ):
        raise ValueError("baseline ranking identity differs")
    _exact_integer(baselines.get("document_count"), 8_114, "baseline documents")
    rankings = _object(baselines.get("rankings"), "baseline ranking inventory")
    if set(rankings) != {"NARRATIVE", "FIXED-O0"}:
        raise ValueError("baseline arm inventory differs")
    for arm in ("NARRATIVE", "FIXED-O0"):
        binding = _object(rankings.get(arm), f"{arm} ranking")
        _exact_integer(binding.get("rows"), 8_114, f"{arm} ranking rows")
        _sha256(binding.get("sha256"), f"{arm} ranking SHA-256")
    _exact_zero(baselines.get("model_load_count"), "baseline model load count")
    _exact_zero(baselines.get("inference_count"), "baseline inference count")

    if (
        r1.get("schema_version")
        != "adaptive-obligation-v2-proposal-incident-r1"
        or r1.get("status") != "aborted"
        or r1.get("reason_code") != "scope_rationale_length_exceeded"
    ):
        raise ValueError("R1 incident identity differs")
    for field, expected in (
        ("attempted_job_count", 1),
        ("terminal_schema_error_count", 1),
        ("uncalled_job_count", 47),
        ("observed_rationale_characters", 245),
        ("accepted_rationale_maximum", 240),
        ("output_token_count", 186),
    ):
        _exact_integer(r1.get(field), expected, f"R1 {field}")
    r1_preflight = _object(r1.get("preflight"), "R1 preflight binding")
    r1_approval = _object(r1.get("approval"), "R1 approval binding")
    r1_ledger = _object(r1.get("ledger"), "R1 ledger binding")
    r1_raw = _object(r1.get("raw_completion"), "R1 raw binding")
    _exact_integer(r1_ledger.get("event_count"), 2, "R1 ledger events")
    _exact_integer(r1_raw.get("bytes"), 546, "R1 raw bytes")
    for binding, label in (
        (r1_preflight, "R1 preflight"),
        (r1_approval, "R1 approval"),
        (_object(r1_ledger.get("anchor"), "R1 anchor"), "R1 anchor"),
        (_object(r1_ledger.get("events"), "R1 events"), "R1 events"),
        (_object(r1_ledger.get("head"), "R1 head"), "R1 head"),
        (r1_raw, "R1 raw completion"),
    ):
        _sha256(binding.get("sha256"), f"{label} SHA-256")

    if (
        r2.get("schema_version")
        != "adaptive-obligation-v2-proposal-preflight-r2"
        or r2.get("status") != "complete"
        or r2.get("prompt_revision") != "tail-contract-r2"
        or r2.get("model") != MODEL_ID
        or r2.get("model_revision") != MODEL_REVISION
        or r2.get("generation_allowed") is not False
        or r2.get("model_construction_allowed") is not False
    ):
        raise ValueError("R2 preflight identity differs")
    for field, expected in (
        ("job_count", 48),
        ("primary_call_count", 48),
        ("retry_call_ceiling", 48),
        ("worst_case_call_ceiling", 96),
        ("primary_max_new_tokens", 256),
        ("retry_max_new_tokens", 512),
        ("tokenizer_load_count", 1),
        ("model_load_count", 0),
        ("inference_count", 0),
    ):
        _exact_integer(r2.get(field), expected, f"R2 {field}")
    prompt_tokens = _object(r2.get("prompt_token_counts"), "R2 prompt tokens")
    for field, expected in (
        ("count", 48),
        ("minimum", 33_610),
        ("maximum", 64_264),
        ("total", 2_300_662),
    ):
        _exact_integer(prompt_tokens.get(field), expected, f"R2 prompt tokens {field}")
    for field in ("prompt_sha256", "schema_sha256", "tokenizer_identity_sha256"):
        _sha256(r2.get(field), f"R2 {field}")
    model_snapshot = _object(r2.get("model_snapshot"), "R2 model snapshot")
    _sha256(model_snapshot.get("manifest_sha256"), "R2 model manifest SHA-256")
    code_hashes = _object(r2.get("code_sha256"), "R2 code hashes")
    if set(code_hashes) != {
        "adaptive_obligation_v2_contract.py",
        "adaptive_obligation_v2_propose.py",
        "adaptive_obligation_v2_propose_r2.py",
    }:
        raise ValueError("R2 code hash inventory differs")
    for name, digest in code_hashes.items():
        _sha256(digest, f"R2 {name} SHA-256")
    r2_artifacts = _object(r2.get("artifacts"), "R2 preflight artifacts")
    if set(r2_artifacts) != {"jobs.jsonl", "prompt.json", "schema.json"}:
        raise ValueError("R2 preflight artifact inventory differs")
    for name, binding_value in r2_artifacts.items():
        binding = _object(binding_value, f"R2 {name} binding")
        _sha256(binding.get("sha256"), f"R2 {name} SHA-256")

    ledger_dir = r2.get("ledger_dir")
    proposal_dir = r2.get("proposal_dir")
    if (
        not isinstance(ledger_dir, str)
        or not Path(ledger_dir).is_absolute()
        or not isinstance(proposal_dir, str)
        or not Path(proposal_dir).is_absolute()
    ):
        raise ValueError("R2 frozen destinations differ")

    paths = {
        name: _safe_repo_path(source_paths.get(name), f"{name} source path")
        for name in SOURCE_NAMES
    }
    hashes = {
        name: _sha256(source_hashes.get(name), f"{name} receipt SHA-256")
        for name in SOURCE_NAMES
    }
    if set(supporting_paths) != set(SUPPORTING_SOURCE_PATHS) or set(
        supporting_hashes
    ) != set(SUPPORTING_SOURCE_PATHS):
        raise ValueError("report supporting source inventory differs")
    checked_supporting_paths = {
        name: _safe_repo_path(
            supporting_paths.get(name), f"{name} supporting source path"
        )
        for name in SUPPORTING_SOURCE_PATHS
    }
    checked_supporting_hashes = {
        name: _sha256(
            supporting_hashes.get(name), f"{name} supporting source SHA-256"
        )
        for name in SUPPORTING_SOURCE_PATHS
    }

    return {
        "contract": contract,
        "baselines": baselines,
        "r1": r1,
        "r2": r2,
        "rankings": rankings,
        "r1_preflight": r1_preflight,
        "r1_approval": r1_approval,
        "r1_ledger": r1_ledger,
        "r1_raw": r1_raw,
        "prompt_tokens": prompt_tokens,
        "model_snapshot": model_snapshot,
        "code_hashes": code_hashes,
        "r2_artifacts": r2_artifacts,
        "ledger_dir": ledger_dir,
        "proposal_dir": proposal_dir,
        "source_paths": paths,
        "source_hashes": hashes,
        "supporting_paths": checked_supporting_paths,
        "supporting_hashes": checked_supporting_hashes,
    }


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if value is True:
        return "TRUE"
    if value is False:
        return "FALSE"
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _source_specs(
    values: Mapping[str, object],
    *,
    stage_rows: Sequence[Mapping[str, object]],
    proposal_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    paths = _object(values.get("source_paths"), "verified source paths")
    hashes = _object(values.get("source_hashes"), "verified source hashes")
    supporting_paths = _object(
        values.get("supporting_paths"), "verified supporting paths"
    )
    supporting_hashes = _object(
        values.get("supporting_hashes"), "verified supporting hashes"
    )
    receipt_fields = {
        "contract": (
            "status, topic_ids, parent_count, reservoir_count, unit_count, "
            "protected_topic_count, qrels_opened"
        ),
        "baseline_rankings": "status, topic_ids, document_count, rankings, qrels_opened",
        "r1_incident": (
            "status, reason_code, attempted_job_count, terminal_schema_error_count, "
            "uncalled_job_count, output_token_count, qrels_opened"
        ),
        "r2_preflight": (
            "status, topic_ids, job_count, primary_call_count, retry_call_ceiling, "
            "worst_case_call_ceiling, prompt_token_counts, model, model_revision, "
            "ledger_dir, model_load_count, inference_count, qrels_opened"
        ),
    }
    sources: list[dict[str, object]] = []
    for name in SOURCE_NAMES:
        path = str(paths[name])
        digest = str(hashes[name])
        quoted = path.replace("'", "''")
        sources.append(
            {
                "id": name,
                "label": SOURCE_LABELS[name],
                "path": path,
                "query": {
                    "engine": "duckdb",
                    "id": f"sha256:{digest}",
                    "language": "sql",
                    "description": (
                        "Reads the canonical receipt after its repository verifier "
                        "authenticated the complete bound artifact."
                    ),
                    "sql": (
                        f"SELECT {receipt_fields[name]} "
                        f"FROM read_json_auto('{quoted}');"
                    ),
                    "tables_used": [path],
                    "filters": [
                        "Pilot topics 219, 72, 300, and 84 only",
                        "Protected topics 144, 213, 224, 407, and 515 excluded",
                        "No qrels opened",
                    ],
                    "metric_definitions": [
                        f"Receipt identity is exact file SHA-256 {digest}."
                    ],
                },
            }
        )

    r1_preflight = _object(values.get("r1_preflight"), "R1 preflight binding")
    r1_preflight_path = _safe_repo_path(
        r1_preflight.get("path"), "R1 preflight source path"
    )
    r1_preflight_sha = _sha256(
        r1_preflight.get("sha256"), "R1 preflight source SHA-256"
    )
    sources.append(
        {
            "id": "r1_preflight",
            "label": "Incident-authenticated proposal R1 preflight receipt",
            "path": r1_preflight_path,
            "query": {
                "engine": "duckdb",
                "id": f"sha256:{r1_preflight_sha}",
                "language": "sql",
                "description": (
                    "The immutable R1 incident verifier replays and authenticates this "
                    "preflight before accepting the incident receipt."
                ),
                "sql": (
                    "SELECT prompt_token_counts, model, model_revision "
                    "FROM read_json_auto('"
                    + r1_preflight_path.replace("'", "''")
                    + "');"
                ),
                "tables_used": [r1_preflight_path],
                "filters": ["First attempted R1 job only for the observed-call basis"],
                "metric_definitions": [
                    f"Receipt identity is exact file SHA-256 {r1_preflight_sha}."
                ],
            },
        }
    )

    support_tables = [str(supporting_paths[name]) for name in SUPPORTING_SOURCE_PATHS]
    support_digest = _sha256_bytes(
        _canonical_bytes(dict(supporting_hashes))
    )
    support_values = ", ".join(
        "(" + ", ".join((_sql_literal(name), _sql_literal(supporting_paths[name]), _sql_literal(supporting_hashes[name]))) + ")"
        for name in SUPPORTING_SOURCE_PATHS
    )
    sources.append(
        {
            "id": "recovery_contract",
            "label": "Hashed R2 recovery plan, design, and implementations",
            "path": str(supporting_paths["recovery_design"]),
            "query": {
                "engine": "duckdb",
                "id": f"sha256:{support_digest}",
                "language": "sql",
                "description": (
                    "Exact implementation and operator-estimate claims transcribed "
                    "from the hashed recovery sources."
                ),
                "sql": (
                    "SELECT * FROM (VALUES "
                    + support_values
                    + ") AS recovery_sources(name, path, sha256);"
                ),
                "tables_used": support_tables,
                "filters": [
                    "R2 recovery scope only",
                    "Runtime is an operator estimate, not a receipt metric",
                    "Validation integration remains deferred",
                ],
                "metric_definitions": [
                    "The 2–3 hour estimate is based on the observed approximately "
                    "three-minute, 54,402-token R1 call."
                ],
            },
        }
    )

    inventory_values = []
    for row in stage_rows:
        inventory_values.append(
            "("
            + ", ".join(
                map(
                    _sql_literal,
                    (
                        "stage",
                        row["order"],
                        row["stage"],
                        row["status"],
                        row["evidence"],
                        row["calls_completed"],
                        row["canonical_artifact_ready"],
                        None,
                        None,
                    ),
                )
            )
            + ")"
        )
    for row in proposal_rows:
        inventory_values.append(
            "("
            + ", ".join(
                map(
                    _sql_literal,
                    (
                        "proposal_run",
                        None,
                        row["run"],
                        row["status"],
                        row["evidence"],
                        row["calls_completed"],
                        None,
                        row["preflight_sha256"],
                        row["r2_calls_completed"],
                    ),
                )
            )
            + ")"
        )
    inventory_sql = (
        "SELECT * FROM (VALUES "
        + ", ".join(inventory_values)
        + ") AS verified_status(record_type, \"order\", stage, status, evidence, "
        "calls_completed, canonical_artifact_ready, preflight_sha256, "
        "r2_calls_completed);"
    )
    inventory_material = {
        "source_hashes": dict(hashes),
        "supporting_hashes": dict(supporting_hashes),
        "stage_rows": list(stage_rows),
        "proposal_rows": list(proposal_rows),
    }
    inventory_digest = _sha256_bytes(_canonical_bytes(inventory_material))
    sources.append(
        {
            "id": "verified_status_inventory",
            "label": "Verified R1/R2 status and source inventory",
            "path": str(supporting_paths["report_builder"]),
            "query": {
                "engine": "duckdb",
                "id": f"sha256:{inventory_digest}",
                "language": "sql",
                "description": (
                    "Reproduces the reviewed chart and status-table rows from the four "
                    "verified receipts and hashed implementation sources."
                ),
                "sql": inventory_sql,
                "tables_used": [str(paths[name]) for name in SOURCE_NAMES]
                + support_tables,
                "filters": [
                    "R2 approval absent",
                    "R2 ledger absent",
                    "R2 proposals absent",
                    "No model inference, validation, BM25, MiniLM, or qrels",
                ],
                "metric_definitions": [
                    "canonical_artifact_ready=1 only when the named frozen artifact exists and verifies.",
                    "calls_completed counts observed calls; planned call ceilings are separate fields.",
                ],
            },
        }
    )
    return sources


def build_r2_report_artifact(
    sources: Mapping[str, object], *, build_timestamp: str | None = None
) -> dict[str, object]:
    """Reduce authenticated evidence to one canonical portable report artifact."""

    values = _verified_report_values(sources)
    generated_at = _validated_build_timestamp(
        build_timestamp if build_timestamp is not None else _utc_build_timestamp()
    )
    r1 = _object(values["r1"], "R1 incident")
    r2 = _object(values["r2"], "R2 preflight")
    prompt_tokens = _object(values["prompt_tokens"], "R2 prompt tokens")
    source_hashes = _object(values["source_hashes"], "source hashes")
    supporting_paths = _object(values["supporting_paths"], "supporting paths")
    supporting_hashes = _object(values["supporting_hashes"], "supporting hashes")
    r1_preflight = _object(values["r1_preflight"], "R1 preflight")
    r1_approval = _object(values["r1_approval"], "R1 approval")
    r1_ledger = _object(values["r1_ledger"], "R1 ledger")
    r1_raw = _object(values["r1_raw"], "R1 raw")
    r2_artifacts = _object(values["r2_artifacts"], "R2 artifacts")
    code_hashes = _object(values["code_hashes"], "R2 code hashes")
    model_snapshot = _object(values["model_snapshot"], "R2 model snapshot")

    proposal_rows = [
        {
            "run": "R1",
            "status": "aborted after terminal schema error",
            "evidence": "One call attempted; rationale 245 characters versus accepted maximum 240",
            "preflight_sha256": str(r1_preflight["sha256"]),
            "calls_completed": 1,
            "r2_calls_completed": 0,
            "primary_call_count": None,
            "retry_call_ceiling": None,
            "worst_case_call_ceiling": None,
            "prompt_tokens_minimum": None,
            "prompt_tokens_maximum": None,
            "prompt_tokens_total": None,
            "model": None,
            "model_revision": None,
            "ledger_dir": str(r1_ledger["path"]),
        },
        {
            "run": "R2",
            "status": "preflight ready; inference not run",
            "evidence": "Tokenizer-only preflight verified; approval, ledger, and proposals absent",
            "preflight_sha256": str(source_hashes["r2_preflight"]),
            "calls_completed": 0,
            "r2_calls_completed": 0,
            "primary_call_count": 48,
            "retry_call_ceiling": 48,
            "worst_case_call_ceiling": 96,
            "prompt_tokens_minimum": int(prompt_tokens["minimum"]),
            "prompt_tokens_maximum": int(prompt_tokens["maximum"]),
            "prompt_tokens_total": int(prompt_tokens["total"]),
            "model": str(r2["model"]),
            "model_revision": str(r2["model_revision"]),
            "ledger_dir": str(values["ledger_dir"]),
        },
    ]
    stage_rows = [
        {
            "order": 1,
            "stage": "Sealed baselines",
            "status": "sealed",
            "evidence": "NARRATIVE and FIXED-O0; 8,114 rows each",
            "calls_completed": 0,
            "canonical_artifact_ready": 1,
        },
        {
            "order": 2,
            "stage": "R1 incident record",
            "status": "verified aborted run",
            "evidence": "Exactly one schema-invalid call; 47 jobs uncalled",
            "calls_completed": 1,
            "canonical_artifact_ready": 1,
        },
        {
            "order": 3,
            "stage": "R2 proposal preflight",
            "status": "ready",
            "evidence": "48 jobs; tokenizer-only; exact approval boundary frozen",
            "calls_completed": 0,
            "canonical_artifact_ready": 1,
        },
        {
            "order": 4,
            "stage": "R2 proposal inference",
            "status": "not run",
            "evidence": "Approval, ledger, and proposals absent",
            "calls_completed": 0,
            "canonical_artifact_ready": 0,
        },
        {
            "order": 5,
            "stage": "R2 validation integration",
            "status": "unavailable and deferred",
            "evidence": "Separate end-to-end revision/provenance plan required",
            "calls_completed": 0,
            "canonical_artifact_ready": 0,
        },
        {
            "order": 6,
            "stage": "Focused BM25 retrieval",
            "status": "not run",
            "evidence": "No authenticated R2 proposal inventory can enter retrieval",
            "calls_completed": 0,
            "canonical_artifact_ready": 0,
        },
        {
            "order": 7,
            "stage": "MiniLM reranking",
            "status": "not run",
            "evidence": "No R2 retrieval candidates or scoring preflight",
            "calls_completed": 0,
            "canonical_artifact_ready": 0,
        },
        {
            "order": 8,
            "stage": "Qrels evaluation",
            "status": "unopened",
            "evidence": "No adaptive ranking; qrels remain unopened",
            "calls_completed": 0,
            "canonical_artifact_ready": 0,
        },
    ]
    r1_metrics = [
        {
            "calls_completed": 1,
            "uncalled_jobs": 47,
            "output_tokens": 186,
            "raw_bytes": 546,
        }
    ]
    r2_metrics = [
        {
            "calls_completed": 0,
            "primary_calls": 48,
            "retry_ceiling": 48,
            "worst_case_calls": 96,
            "prompt_tokens_total": int(prompt_tokens["total"]),
        }
    ]
    runtime_estimate = [
        {
            "basis": "approximately three-minute, 54,402-token R1 call",
            "estimate_low_hours": 2,
            "estimate_high_hours": 3,
            "receipt_verified": False,
        }
    ]
    r2_contract_rows = [
        {
            "prompt_revision": "tail-contract-r2",
            "accepted_label_max_characters": 120,
            "accepted_rationale_max_characters": 240,
            "target_label_max_characters": 80,
            "target_label_max_words": 10,
            "target_rationale_exact_sentences": 1,
            "target_rationale_max_characters": 160,
            "target_rationale_max_words": 25,
            "primary_max_new_tokens": 256,
            "retry_max_new_tokens": 512,
            "retry_rule": "One retry only for incomplete JSON at the 256-token ceiling",
        }
    ]

    r1_hash_rows = [
        {
            "artifact": "R1 incident receipt",
            "sha256": str(source_hashes["r1_incident"]),
        },
        {"artifact": "R1 preflight receipt", "sha256": str(r1_preflight["sha256"])},
        {"artifact": "R1 approval", "sha256": str(r1_approval["sha256"])},
        {
            "artifact": "R1 ledger anchor",
            "sha256": str(_object(r1_ledger["anchor"], "R1 anchor")["sha256"]),
        },
        {
            "artifact": "R1 ledger events",
            "sha256": str(_object(r1_ledger["events"], "R1 events")["sha256"]),
        },
        {
            "artifact": "R1 ledger head",
            "sha256": str(_object(r1_ledger["head"], "R1 head")["sha256"]),
        },
        {"artifact": "R1 raw completion", "sha256": str(r1_raw["sha256"])},
    ]
    r2_hash_rows = [
        {
            "artifact": "R2 preflight receipt",
            "sha256": str(source_hashes["r2_preflight"]),
        },
        {"artifact": "R2 prompt contract", "sha256": str(r2["prompt_sha256"])},
        {"artifact": "R2 accepted schema", "sha256": str(r2["schema_sha256"])},
        {
            "artifact": "R2 tokenizer identity",
            "sha256": str(r2["tokenizer_identity_sha256"]),
        },
        {
            "artifact": "R2 model snapshot manifest",
            "sha256": str(model_snapshot["manifest_sha256"]),
        },
    ]
    for name in ("jobs.jsonl", "prompt.json", "schema.json"):
        r2_hash_rows.append(
            {
                "artifact": f"R2 {name}",
                "sha256": str(_object(r2_artifacts[name], name)["sha256"]),
            }
        )
    for name, digest in code_hashes.items():
        r2_hash_rows.append({"artifact": f"Frozen code: {name}", "sha256": str(digest)})
    implementation_hash_rows = [
        {
            "source": name,
            "path": str(supporting_paths[name]),
            "sha256": str(supporting_hashes[name]),
        }
        for name in SUPPORTING_SOURCE_PATHS
    ]
    source_receipt_rows = [
        {
            "source": SOURCE_LABELS[name],
            "path": str(_object(values["source_paths"], "paths")[name]),
            "sha256": str(source_hashes[name]),
            "verification": "passed",
        }
        for name in SOURCE_NAMES
    ]

    artifact_sources = _source_specs(
        values, stage_rows=stage_rows, proposal_rows=proposal_rows
    )
    cards = [
        {
            "id": "r1_calls",
            "dataset": "r1_metrics",
            "sourceId": "r1_incident",
            "description": "Observed R1 calls before the terminal schema error.",
            "metrics": [
                {
                    "field": "calls_completed",
                    "label": "R1 calls completed",
                    "format": "number",
                }
            ],
        },
        {
            "id": "r2_calls",
            "dataset": "r2_metrics",
            "sourceId": "r2_preflight",
            "description": "Observed R2 inference calls; planned calls are separate.",
            "metrics": [
                {
                    "field": "calls_completed",
                    "label": "R2 calls completed",
                    "format": "number",
                }
            ],
        },
        {
            "id": "r2_primary_calls",
            "dataset": "r2_metrics",
            "sourceId": "r2_preflight",
            "description": "Frozen primary-call count awaiting exact R2 approval.",
            "metrics": [
                {
                    "field": "primary_calls",
                    "label": "R2 primary calls planned",
                    "format": "number",
                }
            ],
        },
        {
            "id": "r2_worst_case_calls",
            "dataset": "r2_metrics",
            "sourceId": "r2_preflight",
            "description": "Primary calls plus the truncation-only retry ceiling.",
            "metrics": [
                {
                    "field": "worst_case_calls",
                    "label": "R2 worst-case call ceiling",
                    "format": "number",
                }
            ],
        },
        {
            "id": "r2_prompt_tokens",
            "dataset": "r2_metrics",
            "sourceId": "r2_preflight",
            "description": "Exact total prompt tokens across all 48 frozen jobs.",
            "metrics": [
                {
                    "field": "prompt_tokens_total",
                    "label": "R2 prompt tokens total",
                    "format": "number",
                }
            ],
        },
    ]
    charts = [
        {
            "id": "stage_readiness",
            "title": "Canonical artifact readiness by stage",
            "subtitle": (
                "Binary audit state: 1 = named artifact exists and verifies; "
                "0 = not run or unavailable. Not relevance quality or progress."
            ),
            "type": "horizontalBar",
            "dataset": "stage_readiness",
            "sourceId": "verified_status_inventory",
            "valueFormat": "number",
            "layout": "full",
            "encodings": {
                "x": {"field": "stage", "type": "nominal", "label": "Stage"},
                "y": {
                    "field": "canonical_artifact_ready",
                    "type": "quantitative",
                    "label": "Canonical artifact ready (1=yes, 0=no)",
                    "format": "number",
                },
                "label": {
                    "field": "canonical_artifact_ready",
                    "type": "quantitative",
                    "label": "Readiness",
                },
                "tooltip": [
                    {"field": "status", "type": "nominal", "label": "Status"},
                    {
                        "field": "calls_completed",
                        "type": "quantitative",
                        "label": "Calls completed",
                    },
                    {"field": "evidence", "type": "text", "label": "Evidence"},
                ],
            },
            "settings": {
                "orientation": "horizontal",
                "sort": "custom",
                "showValues": True,
            },
        }
    ]
    tables = [
        {
            "id": "proposal_runs",
            "title": "R1 incident and R2 preflight state",
            "subtitle": "Observed calls are separate from frozen call ceilings.",
            "dataset": "proposal_runs",
            "sourceId": "verified_status_inventory",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {"field": "run", "direction": "asc"},
            "columns": [
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {
                    "field": "calls_completed",
                    "label": "Calls completed",
                    "type": "number",
                },
                {
                    "field": "primary_call_count",
                    "label": "Primary ceiling",
                    "type": "number",
                },
                {
                    "field": "retry_call_ceiling",
                    "label": "Retry ceiling",
                    "type": "number",
                },
                {
                    "field": "worst_case_call_ceiling",
                    "label": "Worst-case ceiling",
                    "type": "number",
                },
                {
                    "field": "prompt_tokens_minimum",
                    "label": "Prompt tokens min",
                    "type": "number",
                },
                {
                    "field": "prompt_tokens_maximum",
                    "label": "Prompt tokens max",
                    "type": "number",
                },
                {
                    "field": "prompt_tokens_total",
                    "label": "Prompt tokens total",
                    "type": "number",
                },
                {
                    "field": "preflight_sha256",
                    "label": "Preflight SHA-256",
                    "type": "text",
                },
            ],
        },
        {
            "id": "pipeline_state",
            "title": "Exact downstream state",
            "subtitle": "Unavailable stages are not zero-quality results.",
            "dataset": "stage_readiness",
            "sourceId": "verified_status_inventory",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "Step", "type": "number"},
                {"field": "stage", "label": "Stage", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {
                    "field": "calls_completed",
                    "label": "Calls completed",
                    "type": "number",
                },
            ],
        },
        {
            "id": "r1_hashes",
            "title": "Immutable R1 hashes",
            "subtitle": "Incident, approval, ledger, and raw-completion identities.",
            "dataset": "r1_hashes",
            "sourceId": "r1_incident",
            "density": "dense",
            "layout": "full",
            "defaultSort": {"field": "artifact", "direction": "asc"},
            "columns": [
                {"field": "artifact", "label": "Artifact", "type": "text"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
            ],
        },
        {
            "id": "r2_hashes",
            "title": "Frozen R2 preflight hashes",
            "subtitle": "Receipt, prompt, schema, tokenizer, model, artifacts, and code.",
            "dataset": "r2_hashes",
            "sourceId": "r2_preflight",
            "density": "dense",
            "layout": "full",
            "defaultSort": {"field": "artifact", "direction": "asc"},
            "columns": [
                {"field": "artifact", "label": "Artifact", "type": "text"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
            ],
        },
        {
            "id": "r2_contract",
            "title": "R2 prompt and retry contract",
            "subtitle": "Generation targets are tighter than the unchanged accepted schema.",
            "dataset": "r2_contract",
            "sourceId": "r2_preflight",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {"field": "prompt_revision", "direction": "asc"},
            "columns": [
                {"field": "prompt_revision", "label": "Prompt revision", "type": "text"},
                {
                    "field": "target_label_max_characters",
                    "label": "Label target chars",
                    "type": "number",
                },
                {
                    "field": "target_rationale_max_characters",
                    "label": "Rationale target chars",
                    "type": "number",
                },
                {
                    "field": "accepted_rationale_max_characters",
                    "label": "Accepted rationale chars",
                    "type": "number",
                },
                {"field": "retry_rule", "label": "Retry rule", "type": "text"},
            ],
        },
        {
            "id": "implementation_hashes",
            "title": "Hashed recovery sources",
            "subtitle": "Exact plan, design, implementation, and report-builder identities.",
            "dataset": "implementation_hashes",
            "sourceId": "recovery_contract",
            "density": "dense",
            "layout": "full",
            "defaultSort": {"field": "source", "direction": "asc"},
            "columns": [
                {"field": "source", "label": "Source", "type": "text"},
                {"field": "path", "label": "Repository path", "type": "text"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
            ],
        },
        {
            "id": "source_receipts",
            "title": "Authenticated report receipts",
            "subtitle": "Exact repository-relative receipt paths and file hashes.",
            "dataset": "source_receipts",
            "sourceId": "verified_status_inventory",
            "density": "dense",
            "layout": "full",
            "defaultSort": {"field": "source", "direction": "asc"},
            "columns": [
                {"field": "source", "label": "Source", "type": "text"},
                {"field": "path", "label": "Repository path", "type": "text"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
                {"field": "verification", "label": "Verification", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Technical summary\n\n"
                "**R1 stopped safely, and R2 is preflight-ready but unrun.** "
                "No adaptive relevance result exists. The report records an incident "
                "and an approval boundary—not an effectiveness comparison."
            ),
        },
        {
            "id": "technical_summary_r1",
            "type": "markdown",
            "layout": "full",
            "sourceId": "r1_incident",
            "body": (
                "**R1 stopped after exactly one schema-invalid call.** The valid JSON "
                "carried a 245-character rationale against the unchanged 240-character "
                "accepted limit; 47 jobs remained uncalled, and the exact raw bytes and "
                "ledger chain remain immutable."
            ),
        },
        {
            "id": "technical_summary_r2",
            "type": "markdown",
            "layout": "full",
            "sourceId": "r2_preflight",
            "body": (
                "**R2 proposal inference has not run.** The tokenizer-only preflight "
                "freezes 48 primary calls, at most 48 truncation retries, a 96-call "
                "worst-case ceiling, and zero completed R2 calls."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "r1_calls",
                "r2_calls",
                "r2_primary_calls",
                "r2_worst_case_calls",
                "r2_prompt_tokens",
            ],
        },
        {
            "id": "key_findings",
            "type": "markdown",
            "layout": "full",
            "body": "## The evidence separates an immutable incident from unopened execution",
        },
        {
            "id": "r1_incident_evidence",
            "type": "markdown",
            "layout": "full",
            "sourceId": "r1_incident",
            "body": (
                "**The R1 failure is terminal, narrow, and preserved.** One attempted job "
                "ended in one schema error with 186 output tokens and a 546-byte raw "
                "completion. It was not truncated, repaired, accepted, retried, or sealed."
            ),
        },
        {"id": "r1_hash_table", "type": "table", "tableId": "r1_hashes", "layout": "full"},
        {
            "id": "r2_readiness_evidence",
            "type": "markdown",
            "layout": "full",
            "sourceId": "r2_preflight",
            "body": (
                "**R2 is ready only at the preflight boundary.** Its 48 prompts contain "
                "33,610–64,264 tokens each (2,300,662 total), the pinned model/revision and "
                "tail-adjacent output contract are frozen, and model loads and inference "
                "calls remain zero."
            ),
        },
        {"id": "proposal_run_table", "type": "table", "tableId": "proposal_runs", "layout": "full"},
        {"id": "r2_contract_table", "type": "table", "tableId": "r2_contract", "layout": "full"},
        {"id": "r2_hash_table", "type": "table", "tableId": "r2_hashes", "layout": "full"},
        {
            "id": "stage_interpretation",
            "type": "markdown",
            "layout": "full",
            "sourceId": "verified_status_inventory",
            "body": (
                "**Read readiness as a binary artifact audit, not progress or quality.** "
                "The chart marks only whether each named canonical artifact exists and "
                "verifies. A zero means not run or unavailable; it is not a relevance score."
            ),
        },
        {"id": "stage_chart", "type": "chart", "chartId": "stage_readiness", "layout": "full"},
        {"id": "stage_table", "type": "table", "tableId": "pipeline_state", "layout": "full"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "layout": "full",
            "sourceId": "contract",
            "body": (
                "## Scope and definitions preserve the original evidence boundary\n\n"
                "The pilot covers topics 219, 72, 300, and 84 only: 24 fixed O0 parents, "
                "48 parent/fold reservoirs, and 8,247 exact evidence units. **Preflight "
                "ready** means inputs, hashes, token counts, budgets, and destinations "
                "verify before inference. It does not mean a proposal or relevance result exists."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "layout": "full",
            "sourceId": "recovery_contract",
            "body": (
                "## Recovery changes prompt salience without weakening acceptance\n\n"
                "R2 preserves the accepted 120-character label and 240-character rationale "
                "schema while adding stricter generation targets at the tail of every long "
                "prompt: label at most 80 characters/10 words and one rationale sentence "
                "at most 160 characters/25 words. Only incomplete JSON that reaches the "
                "256-token primary ceiling may receive one 512-token retry."
            ),
        },
        {"id": "implementation_hash_table", "type": "table", "tableId": "implementation_hashes", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "sourceId": "verified_status_inventory",
            "body": (
                "## Limitations keep readiness from becoming an effectiveness claim\n\n"
                "**No adaptive relevance result exists.** R2 proposals, validation, BM25 "
                "retrieval, MiniLM reranking, adaptive rankings, and qrels-backed evaluation "
                "are all absent. No comparison to the sealed baselines is possible."
            ),
        },
        {
            "id": "limitations_validation",
            "type": "markdown",
            "layout": "full",
            "sourceId": "recovery_contract",
            "body": (
                "**R2 validation integration is unavailable and deferred.** R2 proposal "
                "output cannot enter validation until a separate end-to-end revision and "
                "provenance plan versions proposal loading, preflight, approval, execution, "
                "finalization, and receipt verification."
            ),
        },
        {
            "id": "runtime_estimate",
            "type": "markdown",
            "layout": "full",
            "sourceId": "recovery_contract",
            "body": (
                "**Operator estimate: 2–3 hours for the 48 primary R2 calls.** This is "
                "based on the observed approximately three-minute, 54,402-token R1 call. "
                "It is a planning estimate, not a receipt-verified runtime metric, and "
                "truncation retries could extend it."
            ),
        },
        {
            "id": "recommended_next_step",
            "type": "markdown",
            "layout": "full",
            "sourceId": "r2_preflight",
            "body": (
                "## The only next action is exact R2 proposal approval\n\n"
                "**Approve only this exact R2 preflight** SHA-256 `"
                + str(source_hashes["r2_preflight"])
                + "` for model `"
                + MODEL_ID
                + "`, revision `"
                + MODEL_REVISION
                + "`, 48 primary calls, 48 possible truncation retries, a 96-call "
                "worst-case ceiling, and ledger destination `"
                + str(values["ledger_dir"])
                + "`. Do not approve validation, BM25, MiniLM, or qrels."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Questions only after a separately approved R2 proposal run\n\n"
                "- Does the first fail-fast canary produce schema-valid output?\n"
                "- How many of 48 jobs produce supported abstract O1 proposals?\n"
                "- Which separate provenance changes are required before any R2 proposal "
                "can enter validation?"
            ),
        },
        {
            "id": "source_receipt_interpretation",
            "type": "markdown",
            "layout": "full",
            "sourceId": "verified_status_inventory",
            "body": (
                "**Every report claim is anchored to verifier-approved receipts or hashed "
                "recovery sources.** The exact identities below are the reproducibility boundary."
            ),
        },
        {"id": "source_receipt_table", "type": "table", "tableId": "source_receipts", "layout": "full"},
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": (
                "Technical incident/readiness report with one binary artifact chart. "
                "No chart compares heterogeneous counts or implies a relevance result."
            ),
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [
                {
                    "id": source["id"],
                    "label": source["label"],
                    "path": source["path"],
                }
                for source in artifact_sources
            ],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "partial",
            "accessIssues": [
                {
                    "id": "adaptive_result_unavailable",
                    "scope": "Adaptive relevance result",
                    "message": (
                        "R2 proposal inference has not run; no adaptive relevance result exists."
                    ),
                },
                {
                    "id": "r2_validation_integration_deferred",
                    "scope": "R2 validation integration",
                    "message": (
                        "R2 validation integration is unavailable and deferred pending a "
                        "separate end-to-end revision/provenance plan."
                    ),
                },
            ],
            "datasets": {
                "r1_metrics": r1_metrics,
                "r2_metrics": r2_metrics,
                "proposal_runs": proposal_rows,
                "runtime_estimate": runtime_estimate,
                "r2_contract": r2_contract_rows,
                "stage_readiness": stage_rows,
                "r1_hashes": r1_hash_rows,
                "r2_hashes": r2_hash_rows,
                "implementation_hashes": implementation_hash_rows,
                "source_receipts": source_receipt_rows,
            },
        },
        "sources": artifact_sources,
    }


def write_artifact_create_only(path: Path, artifact: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(_canonical_bytes(dict(artifact)))
    except FileExistsError as exc:
        raise FileExistsError(f"create-only report artifact already exists: {path}") from exc


def _verified_canonical_renderer() -> Path:
    if not CANONICAL_RENDERER_PATH.is_file():
        raise RuntimeError("canonical portable report packager is unavailable")
    if _sha256_file(CANONICAL_RENDERER_PATH) != CANONICAL_RENDERER_SHA256:
        raise RuntimeError("canonical portable report packager identity differs")
    return CANONICAL_RENDERER_PATH


def _run_canonical_delivery(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "TMPDIR": "/var/tmp"},
    )


def _expected_portable_counts(artifact: Mapping[str, object]) -> dict[str, int]:
    manifest = _object(artifact.get("manifest"), "report manifest")
    cards = {
        card.get("id")
        for card in manifest.get("cards", [])
        if isinstance(card, Mapping) and card.get("id")
    }
    charts = {
        chart.get("id")
        for chart in manifest.get("charts", [])
        if isinstance(chart, Mapping) and chart.get("id")
    }
    tables = {
        table.get("id")
        for table in manifest.get("tables", [])
        if isinstance(table, Mapping) and table.get("id")
    }
    counts = {"blocks": 0, "charts": 0, "html": 0, "metrics": 0, "tables": 0}
    for block in manifest.get("blocks", []):
        if not isinstance(block, Mapping):
            raise RuntimeError("report manifest contains a malformed block")
        if block.get("type") == "metric-strip":
            count = sum(card_id in cards for card_id in block.get("cardIds", []))
            counts["metrics"] += count
            counts["blocks"] += count
            continue
        counts["blocks"] += 1
        if block.get("type") == "chart" and block.get("chartId") in charts:
            counts["charts"] += 1
        if block.get("type") == "table" and block.get("tableId") in tables:
            counts["tables"] += 1
        if block.get("type") == "html":
            counts["html"] += 1
    return counts


def deliver_html_create_only(
    *, artifact_path: Path, output_path: Path
) -> dict[str, object]:
    """Invoke the canonical portable builder exactly once and publish its output."""

    artifact_path = Path(artifact_path)
    output_path = Path(output_path)
    if _present(output_path):
        raise FileExistsError(f"create-only HTML report already exists: {output_path}")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("canonical report artifact is unreadable") from exc
    if not isinstance(artifact, Mapping):
        raise RuntimeError("canonical report artifact must be an object")
    manifest = _object(artifact.get("manifest"), "report manifest")
    if artifact.get("surface") != "report" or manifest.get("surface") != "report":
        raise RuntimeError("canonical report artifact surface differs")
    expected_counts = _expected_portable_counts(artifact)
    renderer_path = _verified_canonical_renderer()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        completed = _run_canonical_delivery(
            [
                "node",
                str(renderer_path),
                "--input",
                str(artifact_path),
                "--output",
                str(temporary_path),
            ]
        )
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError("portable packager did not produce a non-empty HTML file")
        try:
            receipt = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("portable packager receipt is invalid") from exc
        if not isinstance(receipt, dict):
            raise RuntimeError("portable packager receipt must be an object")
        stages = receipt.get("stages")
        if (
            receipt.get("ok") is not True
            or not isinstance(stages, Mapping)
            or stages.get("validation") != "passed"
            or stages.get("package") != "passed"
            or stages.get("verification") != "passed"
            or receipt.get("counts") != expected_counts
            or receipt.get("html") != str(temporary_path.resolve())
        ):
            raise RuntimeError("portable packager receipt does not match verified report")
        os.link(temporary_path, output_path)
        return receipt
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--baseline-rankings", type=Path, required=True)
    parser.add_argument("--r1-incident", type=Path, required=True)
    parser.add_argument("--r2-preflight", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact_path = args.artifact or args.output.with_name("artifact.json")
    if _present(artifact_path):
        raise FileExistsError(
            f"create-only report artifact already exists: {artifact_path}"
        )
    if _present(args.output):
        raise FileExistsError(f"create-only HTML report already exists: {args.output}")
    verified = load_verified_sources(
        contract_dir=args.contract,
        baseline_rankings_dir=args.baseline_rankings,
        r1_incident_dir=args.r1_incident,
        r2_preflight_dir=args.r2_preflight,
    )
    artifact = build_r2_report_artifact(verified)
    write_artifact_create_only(artifact_path, artifact)
    receipt = deliver_html_create_only(
        artifact_path=artifact_path,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "output": str(args.output),
                "status": "r2_preflight_ready_unrun",
                "delivery": receipt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
