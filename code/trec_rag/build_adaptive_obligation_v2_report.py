"""Build the canonical portable status report for adaptive obligation search v2.

The builder authenticates only already-frozen artifacts.  It never opens qrels,
constructs a model, performs inference, inspects a retrieval cache, or creates a
retrieval request.  ``report.html`` is generated only by the shared Data
Analytics portable-artifact packager from the canonical ``artifact.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import subprocess
import tempfile

from .adaptive_evidence_discovery import verify_discovery_terminal
from .adaptive_evidence_rank import verify_baseline_rankings
from .adaptive_obligation_v2_contract import verify_v2_contract
from .adaptive_obligation_v2_propose import verify_proposal_preflight


TITLE = "Adaptive obligation search v2"
STATUS = "proposal_preflight_ready"
GENERATED_AT = "2026-07-15T00:00:00Z"
PILOT_TOPIC_IDS = ("219", "72", "300", "84")
RETRIEVAL_HITS = 1_000
MAX_RETRIEVAL_REQUESTS = 16
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_BUILDER_PATH = "code/trec_rag/build_adaptive_obligation_v2_report.py"
SOURCE_NAMES = (
    "contract",
    "proposal_preflight",
    "baseline_rankings",
    "v1_discovery",
)
SOURCE_LABELS = {
    "contract": "Verified v2 evidence contract receipt",
    "proposal_preflight": "Verified v2 proposal preflight receipt",
    "baseline_rankings": "Sealed NARRATIVE and FIXED-O0 ranking receipt",
    "v1_discovery": "Terminal discovery v1 receipt",
}
EXPECTED_V2_ROOT_CHILDREN = frozenset({"contract", "proposal_preflight"})


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
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise ValueError(f"{label} must be exact integer zero")


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _repo_relative(path: Path) -> str:
    resolved = Path(path).resolve(strict=True)
    try:
        return resolved.relative_to(REPO_ROOT.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ValueError(f"report source is outside the repository: {path}") from exc


def _present(path: Path) -> bool:
    return os.path.lexists(path)


def _assert_inference_free_v2_root(v2_root: Path) -> None:
    try:
        observed = {entry.name for entry in v2_root.iterdir()}
    except OSError as exc:
        raise ValueError("v2 output root is unreadable") from exc
    unexpected = observed - EXPECTED_V2_ROOT_CHILDREN
    missing = EXPECTED_V2_ROOT_CHILDREN - observed
    if unexpected or missing:
        detail = ", ".join(sorted(unexpected | missing))
        raise ValueError(f"v2 later artifact boundary differs: {detail}")
    if any("approval" in entry.name.casefold() for entry in v2_root.rglob("*")):
        raise ValueError("v2 approval artifact exists")


def load_verified_sources(
    *,
    contract_dir: Path,
    proposal_preflight_dir: Path,
    baseline_rankings_dir: Path,
    v1_discovery_dir: Path,
) -> dict[str, object]:
    """Authenticate the four canonical inputs and prove later v2 outputs absent."""

    contract_dir = Path(contract_dir)
    proposal_preflight_dir = Path(proposal_preflight_dir)
    baseline_rankings_dir = Path(baseline_rankings_dir)
    v1_discovery_dir = Path(v1_discovery_dir)
    if contract_dir.parent != proposal_preflight_dir.parent:
        raise ValueError("v2 contract and proposal preflight roots differ")
    _assert_inference_free_v2_root(contract_dir.parent)

    receipts = {
        "contract": verify_v2_contract(contract_dir),
        "proposal_preflight": verify_proposal_preflight(proposal_preflight_dir),
        "baseline_rankings": verify_baseline_rankings(baseline_rankings_dir),
        "v1_discovery": verify_discovery_terminal(v1_discovery_dir),
    }
    receipt_paths = {
        "contract": contract_dir / "receipt.json",
        "proposal_preflight": proposal_preflight_dir / "receipt.json",
        "baseline_rankings": baseline_rankings_dir / "receipt.json",
        "v1_discovery": v1_discovery_dir / "receipt.json",
    }
    return {
        **receipts,
        "source_paths": {
            name: _repo_relative(path) for name, path in receipt_paths.items()
        },
        "source_hashes": {
            name: _sha256_file(path) for name, path in receipt_paths.items()
        },
        "later_artifacts_present": [],
        "approval_present": False,
    }


def _verify_common_safety(receipt: Mapping[str, object], label: str) -> None:
    if receipt.get("qrels_opened") is not False:
        raise ValueError(f"{label} qrels state differs")
    for field in (
        "network_call_count",
        "retrieval_call_count",
        "hosted_inference_call_count",
        "paid_call_count",
    ):
        _exact_zero(receipt.get(field), f"{label} {field}")
    if receipt.get("external_cost_usd") != 0.0:
        raise ValueError(f"{label} external cost differs")


def build_report_payload(sources: Mapping[str, object]) -> dict[str, object]:
    """Reduce authenticated receipts to the bounded reader-facing report data."""

    contract = _object(sources.get("contract"), "v2 contract")
    proposal = _object(sources.get("proposal_preflight"), "proposal preflight")
    baselines = _object(sources.get("baseline_rankings"), "baseline rankings")
    terminal_v1 = _object(sources.get("v1_discovery"), "terminal v1 receipt")
    source_paths = _object(sources.get("source_paths"), "report source paths")
    source_hashes = _object(sources.get("source_hashes"), "report source hashes")

    if any(
        receipt.get("topic_ids") != list(PILOT_TOPIC_IDS)
        for receipt in (contract, proposal, baselines)
    ):
        raise ValueError("report topic boundary differs")
    if sources.get("later_artifacts_present") != []:
        raise ValueError("later adaptive artifacts must be absent")
    if sources.get("approval_present") is not False:
        raise ValueError("v2 approval artifact must be absent")

    for label, receipt in (
        ("v2 contract", contract),
        ("proposal preflight", proposal),
        ("baseline rankings", baselines),
        ("terminal v1", terminal_v1),
    ):
        _verify_common_safety(receipt, label)

    _exact_integer(contract.get("parent_count"), 24, "parent count")
    _exact_integer(contract.get("reservoir_count"), 48, "reservoir count")
    _exact_integer(contract.get("unit_count"), 8_247, "unit count")
    _exact_zero(contract.get("protected_topic_count"), "protected topic count")
    _exact_zero(contract.get("model_load_count"), "contract model load count")
    _exact_zero(contract.get("inference_count"), "contract inference count")

    _exact_integer(proposal.get("job_count"), 48, "proposal job count")
    _exact_integer(proposal.get("primary_call_count"), 48, "proposal primary calls")
    _exact_integer(proposal.get("retry_call_ceiling"), 48, "proposal retry ceiling")
    _exact_integer(
        proposal.get("worst_case_call_ceiling"), 96, "proposal worst-case calls"
    )
    _exact_integer(proposal.get("tokenizer_load_count"), 1, "tokenizer load count")
    _exact_zero(proposal.get("model_load_count"), "proposal model load count")
    _exact_zero(proposal.get("inference_count"), "proposal inference count")

    _exact_integer(baselines.get("document_count"), 8_114, "baseline documents")
    rankings = _object(baselines.get("rankings"), "baseline ranking inventory")
    if set(rankings) != {"NARRATIVE", "FIXED-O0"}:
        raise ValueError("baseline arm inventory differs")
    for arm in ("NARRATIVE", "FIXED-O0"):
        binding = _object(rankings.get(arm), f"{arm} ranking binding")
        _exact_integer(binding.get("rows"), 8_114, f"{arm} row count")
        _sha256(binding.get("sha256"), f"{arm} ranking SHA-256")
    _exact_zero(baselines.get("model_load_count"), "baseline model load count")
    _exact_zero(baselines.get("inference_count"), "baseline inference count")

    if terminal_v1.get("status") != "discovery_unavailable":
        raise ValueError("terminal v1 status differs")
    if terminal_v1.get("reason") != "corrected_pass_schema_json_truncation":
        raise ValueError("terminal v1 reason differs")

    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name in SOURCE_NAMES:
        path = source_paths.get(name)
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            raise ValueError(f"{name} report source path must be repo-relative")
        paths[name] = path
        hashes[name] = _sha256(source_hashes.get(name), f"{name} receipt SHA-256")

    return {
        "status": STATUS,
        "topic_ids": list(PILOT_TOPIC_IDS),
        "parents": 24,
        "reservoirs": 48,
        "units": 8_247,
        "baseline_documents": 8_114,
        "baseline_ranking_rows": {"NARRATIVE": 8_114, "FIXED-O0": 8_114},
        "baseline_ranking_hashes": {
            arm: _object(rankings[arm], f"{arm} ranking")["sha256"]
            for arm in ("NARRATIVE", "FIXED-O0")
        },
        "proposal_jobs": 48,
        "proposal_primary_calls": 48,
        "proposal_retry_call_ceiling": 48,
        "proposal_worst_case_calls": 96,
        "retrieval_hits_per_accepted_o1": RETRIEVAL_HITS,
        "maximum_retrieval_requests": MAX_RETRIEVAL_REQUESTS,
        "accepted_o1_count": 0,
        "qwen_calls_completed": 0,
        "validation_calls_completed": 0,
        "retrieval_calls_completed": 0,
        "minilm_calls_completed": 0,
        "qrels_opened": False,
        "v1_status": terminal_v1["status"],
        "v1_reason": terminal_v1["reason"],
        "source_paths": paths,
        "source_hashes": hashes,
        "contract_artifacts": dict(
            _object(contract.get("artifacts"), "contract artifact inventory")
        )
        if "artifacts" in contract
        else {},
        "proposal_artifacts": dict(
            _object(proposal.get("artifacts"), "proposal artifact inventory")
        )
        if "artifacts" in proposal
        else {},
    }


def _source_specs(payload: Mapping[str, object]) -> list[dict[str, object]]:
    paths = _object(payload.get("source_paths"), "payload source paths")
    hashes = _object(payload.get("source_hashes"), "payload source hashes")
    sources: list[dict[str, object]] = []
    for name in SOURCE_NAMES:
        path = str(paths[name])
        digest = str(hashes[name])
        quoted_path = path.replace("'", "''")
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
                        "Reads the canonical receipt only after its repository verifier "
                        "has authenticated the complete bound artifact."
                    ),
                    "sql": f"SELECT * FROM read_json_auto('{quoted_path}');",
                    "tables_used": [path],
                    "filters": [
                        "Pilot topics 219, 72, 300, and 84 only",
                        "Protected topics 144, 213, 224, 407, and 515 excluded",
                        "No qrels or later adaptive output opened",
                    ],
                    "metric_definitions": [
                        f"Receipt identity is the exact file SHA-256 {digest}."
                    ],
                },
            }
        )
    inventory_digest = _sha256_bytes(
        "\n".join(str(hashes[name]) for name in SOURCE_NAMES).encode("ascii")
    )
    inventory_sql = (
        "WITH contract AS (SELECT * FROM read_json_auto('"
        + str(paths["contract"]).replace("'", "''")
        + "')), proposal AS (SELECT * FROM read_json_auto('"
        + str(paths["proposal_preflight"]).replace("'", "''")
        + "')), baselines AS (SELECT * FROM read_json_auto('"
        + str(paths["baseline_rankings"]).replace("'", "''")
        + "')), terminal_v1 AS (SELECT * FROM read_json_auto('"
        + str(paths["v1_discovery"]).replace("'", "''")
        + "')) SELECT contract.parent_count, contract.reservoir_count, "
        "contract.unit_count, proposal.primary_call_count, "
        "proposal.retry_call_ceiling, proposal.worst_case_call_ceiling, "
        "baselines.document_count, terminal_v1.status AS v1_status "
        "FROM contract, proposal, baselines, terminal_v1;"
    )
    sources.append(
        {
            "id": "verified_inventory",
            "label": "Verified four-receipt report inventory",
            "path": REPORT_BUILDER_PATH,
            "query": {
                "engine": "duckdb",
                "id": f"sha256:{inventory_digest}",
                "language": "sql",
                "description": (
                    "Combines only the four independently verified receipts into the "
                    "bounded report snapshot."
                ),
                "sql": inventory_sql,
                "tables_used": [str(paths[name]) for name in SOURCE_NAMES],
                "filters": [
                    "All later adaptive artifacts and approval files absent",
                    "No model, endpoint, cache, qrels, or network access",
                ],
                "metric_definitions": [
                    "Stage calls completed are zero when the prerequisite canonical "
                    "output is absent and the authenticated preflight records zero execution."
                ],
            },
        }
    )
    return sources


def build_report_artifact(payload: Mapping[str, object]) -> dict[str, object]:
    """Build one deterministic canonical report artifact; no HTML is authored here."""

    if payload.get("status") != STATUS:
        raise ValueError("report payload status differs")
    source_paths = _object(payload.get("source_paths"), "payload source paths")
    source_hashes = _object(payload.get("source_hashes"), "payload source hashes")
    for name in SOURCE_NAMES:
        _sha256(source_hashes.get(name), f"{name} source SHA-256")
        path = source_paths.get(name)
        if not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"{name} source path is unsafe")

    stage_rows = [
        {
            "order": 1,
            "stage": "Fixed baselines",
            "status": "sealed",
            "evidence": "NARRATIVE and FIXED-O0; 8,114 rows each",
            "calls_completed": 0,
            "approval": "not required (already frozen)",
        },
        {
            "order": 2,
            "stage": "V2 evidence contract",
            "status": "verified",
            "evidence": "24 O0 parents; 48 fold reservoirs; 8,247 exact units",
            "calls_completed": 0,
            "approval": "not required (deterministic)",
        },
        {
            "order": 3,
            "stage": "O1 proposal preflight",
            "status": "ready",
            "evidence": "48 exact jobs; 48 primary + at most 48 truncation retries",
            "calls_completed": 0,
            "approval": "separate proposal approval required",
        },
        {
            "order": 4,
            "stage": "Qwen O1 proposals",
            "status": "not run",
            "evidence": "No proposal ledger or canonical proposal inventory",
            "calls_completed": 0,
            "approval": "unopened",
        },
        {
            "order": 5,
            "stage": "Opposite-fold validation",
            "status": "unavailable",
            "evidence": "Requires frozen proposals; no validation preflight exists",
            "calls_completed": 0,
            "approval": "unopened",
        },
        {
            "order": 6,
            "stage": "Focused BM25 retrieval",
            "status": "unavailable",
            "evidence": "Design: one hits=1,000 request per accepted O1, maximum 16",
            "calls_completed": 0,
            "approval": "unopened",
        },
        {
            "order": 7,
            "stage": "O1-local MiniLM",
            "status": "unavailable",
            "evidence": "No accepted O1 candidates or scoring preflight",
            "calls_completed": 0,
            "approval": "unopened",
        },
        {
            "order": 8,
            "stage": "Adaptive ranking and evaluation",
            "status": "unavailable",
            "evidence": "No ADAPTIVE-V2 ranking; qrels unopened",
            "calls_completed": 0,
            "approval": "unopened",
        },
    ]
    source_rows = [
        {
            "source": SOURCE_LABELS[name],
            "path": str(source_paths[name]),
            "sha256": str(source_hashes[name]),
            "verification": "passed",
        }
        for name in SOURCE_NAMES
    ]
    contract_metrics = [
        {
            "parents": int(payload["parents"]),
            "reservoirs": int(payload["reservoirs"]),
            "units": int(payload["units"]),
        }
    ]
    proposal_metrics = [
        {
            "primary_jobs": int(payload["proposal_jobs"]),
            "retry_ceiling": int(payload["proposal_retry_call_ceiling"]),
            "worst_case_calls": int(payload["proposal_worst_case_calls"]),
        }
    ]
    baseline_metrics = [{"documents_per_arm": int(payload["baseline_documents"])}]
    readiness_rows = [
        {
            "order": row["order"],
            "stage": row["stage"],
            "canonical_artifact_ready": 1
            if row["status"] in {"sealed", "verified", "ready"}
            else 0,
            "status": row["status"],
            "evidence": row["evidence"],
            "calls_completed": row["calls_completed"],
        }
        for row in stage_rows
    ]

    cards = [
        {
            "id": "baseline_documents",
            "dataset": "baseline_metrics",
            "sourceId": "baseline_rankings",
            "description": "Rows in each sealed baseline arm; not a relevance metric.",
            "metrics": [
                {
                    "field": "documents_per_arm",
                    "label": "Documents per baseline arm",
                    "format": "number",
                }
            ],
        },
        {
            "id": "o0_parents",
            "dataset": "contract_metrics",
            "sourceId": "contract",
            "description": "Fixed explicit obligations across the four pilot topics.",
            "metrics": [
                {"field": "parents", "label": "O0 parents", "format": "number"}
            ],
        },
        {
            "id": "fold_reservoirs",
            "dataset": "contract_metrics",
            "sourceId": "contract",
            "description": "One frozen evidence reservoir per O0 and document fold.",
            "metrics": [
                {
                    "field": "reservoirs",
                    "label": "Fold reservoirs",
                    "format": "number",
                }
            ],
        },
        {
            "id": "evidence_units",
            "dataset": "contract_metrics",
            "sourceId": "contract",
            "description": "Exact sentence/list-item units available to proposal jobs.",
            "metrics": [
                {"field": "units", "label": "Evidence units", "format": "number"}
            ],
        },
        {
            "id": "proposal_primary_jobs",
            "dataset": "proposal_metrics",
            "sourceId": "proposal_preflight",
            "description": "Exactly one proposal job for every parent/fold pair.",
            "metrics": [
                {
                    "field": "primary_jobs",
                    "label": "Primary proposal jobs",
                    "format": "number",
                }
            ],
        },
        {
            "id": "proposal_retry_ceiling",
            "dataset": "proposal_metrics",
            "sourceId": "proposal_preflight",
            "description": "At most one retry per job, only for exact-ceiling JSON truncation.",
            "metrics": [
                {
                    "field": "retry_ceiling",
                    "label": "Proposal retry ceiling",
                    "format": "number",
                }
            ],
        },
        {
            "id": "proposal_worst_case",
            "dataset": "proposal_metrics",
            "sourceId": "proposal_preflight",
            "description": "Maximum proposal calls; actual v2 proposal calls remain zero.",
            "metrics": [
                {
                    "field": "worst_case_calls",
                    "label": "Worst-case proposal calls",
                    "format": "number",
                }
            ],
        },
    ]
    tables = [
        {
            "id": "stage_status",
            "title": "Exact adaptive v2 stage status",
            "subtitle": "Current canonical state; planned ceilings are not completed work.",
            "dataset": "stage_status",
            "sourceId": "verified_inventory",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "Step", "type": "number"},
                {"field": "stage", "label": "Stage", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "evidence", "label": "Verified evidence", "type": "text"},
                {
                    "field": "calls_completed",
                    "label": "V2 calls completed",
                    "type": "number",
                },
                {"field": "approval", "label": "Approval state", "type": "text"},
            ],
        },
        {
            "id": "source_receipts",
            "title": "Authenticated source receipts",
            "subtitle": "Exact repository-relative paths and file SHA-256 identities.",
            "dataset": "source_receipts",
            "sourceId": "verified_inventory",
            "density": "dense",
            "layout": "full",
            "defaultSort": {"field": "source", "direction": "asc"},
            "columns": [
                {"field": "source", "label": "Source", "type": "text"},
                {"field": "path", "label": "Repository path", "type": "text"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
                {
                    "field": "verification",
                    "label": "Verification",
                    "type": "text",
                },
            ],
        },
    ]
    charts = [
        {
            "id": "stage_readiness",
            "title": "Canonical artifact readiness by pipeline stage",
            "subtitle": (
                "Binary audit state: 1 = canonical artifact exists and verifies; "
                "0 = not run or unavailable. This is not relevance quality or progress."
            ),
            "type": "horizontalBar",
            "dataset": "stage_readiness",
            "sourceId": "verified_inventory",
            "valueFormat": "number",
            "layout": "full",
            "encodings": {
                "x": {
                    "field": "stage",
                    "type": "nominal",
                    "label": "Stage",
                },
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
                        "label": "V2 calls completed",
                    },
                    {
                        "field": "evidence",
                        "type": "text",
                        "label": "Verified evidence",
                    },
                ],
            },
            "settings": {
                "orientation": "horizontal",
                "sort": "custom",
                "showValues": True,
            },
        }
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "layout": "full",
            "body": f"# {TITLE}",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Technical summary\n\n"
                "**Current result: the fixed baselines are sealed; adaptive v2 has not run.** "
                "No adaptive relevance result exists yet. The current status is "
                "`proposal_preflight_ready`, which proves execution readiness—not retrieval "
                "or relevance quality.\n\n"
                "**What is ready:** 48 exact proposal jobs and a worst-case 96-call "
                "proposal ceiling. **What needs separate approval:** Qwen proposals, "
                "opposite-fold validation, up to 16 `hits=1000` BM25 requests, O1 "
                "MiniLM scoring, and qrels. All five remain at zero or unopened for v2."
            ),
        },
        {
            "id": "key_evidence",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## The fixed evidence boundary verifies readiness, not effectiveness\n\n"
                "The two metric strips summarize different units—documents, obligations, "
                "reservoirs, evidence units, jobs, and call ceilings. They are shown as "
                "source-backed status cards rather than a count chart because a shared "
                "visual scale would imply a false comparison. The only chart below uses a "
                "single binary readiness definition shared by every pipeline stage."
            ),
        },
        {
            "id": "evidence_metrics",
            "type": "metric-strip",
            "cardIds": [
                "baseline_documents",
                "o0_parents",
                "fold_reservoirs",
                "evidence_units",
            ],
        },
        {
            "id": "proposal_metrics",
            "type": "metric-strip",
            "cardIds": [
                "proposal_primary_jobs",
                "proposal_retry_ceiling",
                "proposal_worst_case",
            ],
        },
        {
            "id": "stage_status_interpretation",
            "type": "markdown",
            "layout": "full",
            "body": (
                "**Read the table as a state machine, not a performance funnel.** Only the "
                "baselines, evidence contract, and tokenizer-only proposal preflight exist. "
                "Every later row is explicitly unavailable rather than silently treated as "
                "a zero-quality result. The binary chart visualizes that artifact boundary; "
                "it does not measure relevance quality, completion percentage, or progress."
            ),
        },
        {
            "id": "stage_readiness_chart",
            "type": "chart",
            "chartId": "stage_readiness",
            "layout": "full",
        },
        {
            "id": "stage_status_table",
            "type": "table",
            "tableId": "stage_status",
            "layout": "full",
        },
        {
            "id": "source_receipt_interpretation",
            "type": "markdown",
            "layout": "full",
            "body": (
                "**Every visible count above is anchored to one of four verifier-approved "
                "receipts.** The exact paths and hashes below are the reproducibility "
                "boundary; machine-specific absolute paths are intentionally excluded."
            ),
        },
        {
            "id": "source_receipt_table",
            "type": "table",
            "tableId": "source_receipts",
            "layout": "full",
        },
        {
            "id": "scope_definitions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Scope and definitions keep facets tied to the user need\n\n"
                "- **O0** is a fixed, explicit information obligation already frozen from "
                "the original narrative.\n"
                "- **O1** is an optional abstract child need proposed from corpus evidence. "
                "It must stay within one O0's subject, population, domain, and relation; it "
                "is not an answer fact.\n"
                "- A **fold** is one of two deterministic document partitions. Proposal "
                "evidence comes from one fold; validation must use the opposite fold and a "
                "different document. This gives evidence independence, not model "
                "independence.\n"
                "- **Raw-first** means an attempt and its exact raw bytes are durably "
                "recorded before parsing or normalization.\n"
                "- **Query-local** means BM25 or MiniLM scores order only candidates from "
                "the query that produced them; raw scores never cross query boundaries."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## The bounded pipeline can add candidates without becoming a free-running agent\n\n"
                "1. Start from the sealed NARRATIVE and FIXED-O0 baselines.\n"
                "2. Use 24 O0 parents × two document folds to form 48 frozen evidence "
                "reservoirs.\n"
                "3. After separate approval, Qwen proposes at most one evidence-supported "
                "O1 per parent/fold job.\n"
                "4. Validate each surviving O1 only against opposite-fold evidence, then "
                "cap acceptance at one O1 per parent and four per topic.\n"
                "5. Render one focused BM25 query as ordered parent anchors + complete O0 "
                "+ accepted O1. One `hits=1000` response supplies both top-100 and "
                "top-1,000 diagnostics.\n"
                "6. In a later approved stage, score each O1 queue with MiniLM using the "
                "unchanged narrative + complete O0 + O1.\n"
                "7. Preserve the complete baseline union, append genuinely new documents, "
                "freeze all rankings, and only then open qrels.\n\n"
                "The BM25 query is intentionally focused for lexical matching; the future "
                "MiniLM query is broader so semantic filtering can judge each candidate "
                "against the original user need and its local facet. There is no recursive "
                "search-generation loop."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Limitations and robustness checks\n\n"
                "- **There is no adaptive effectiveness evidence.** No accepted O1, BM25 "
                "response, O1 MiniLM score, ADAPTIVE-V2 ranking, or qrels-backed metric "
                "exists. Readiness must not be reported as improvement.\n"
                "- **The pilot is bounded to four topics.** Even a later positive result "
                "would be descriptive, not production generalization.\n"
                "- **The hosted retriever is plain text plus `hits`.** Field selection, "
                "required terms, boosts, phrase/slop, RM3, and BM25 parameter changes are "
                "not assumed.\n"
                "- **Discovery v1 remains terminal.** Its historical status is "
                "`discovery_unavailable` after JSON truncation; it is provenance, not an "
                "adaptive result or a mutable v2 input.\n"
                "- **Cross-fold validation is not model independence.** The same pinned "
                "local model may propose and validate, but it sees disjoint evidence and "
                "different prompts.\n"
                "- **Robustness is enforced structurally so far.** Protected-topic checks, "
                "exact hashes, create-only ledgers, finite retries, query-local scoring, "
                "and the qrels firewall are implemented; relevance robustness remains "
                "untested."
            ),
        },
        {
            "id": "recommended_next_step",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Recommended next approval step\n\n"
                "Approve only the frozen proposal stage if you want empirical progress. "
                "The approval must bind proposal receipt SHA-256 `"
                + str(source_hashes["proposal_preflight"])
                + "`, model `Qwen/Qwen3-4B-Instruct-2507`, revision "
                "`cdbee75f17c01a7cc42f958dc650907174af0554`, 48 primary calls, and "
                "a maximum 48 truncation retries.\n\n"
                "After that run seals, inspect supported/unsupported outcomes and freeze a "
                "separate validation preflight. Do not authorize validation, BM25, MiniLM, "
                "or qrels in the same approval."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "- How many of the 48 proposal jobs yield a valid abstract O1 rather than "
                "answer facts, duplicates, or unsupported needs?\n"
                "- Do accepted O1 queries retrieve genuinely new documents, or mainly "
                "rediscover the 8,114-document baseline union?\n"
                "- Does query-local MiniLM keep facet-specific evidence that a single "
                "global narrative score would miss?\n"
                "- If qrels are sparse for newly retrieved documents, what blinded review "
                "is needed before concluding that adaptive recall failed?"
            ),
        },
    ]

    sources = _source_specs(payload)
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": (
                "Technical inference-free readiness report. Counts have heterogeneous "
                "denominators, so no chart compares counts. Source-backed scorecards, a "
                "binary readiness visual, and exact tables avoid a false comparison."
            ),
            "generatedAt": GENERATED_AT,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [
                {
                    "id": source["id"],
                    "label": source["label"],
                    "path": source["path"],
                }
                for source in sources
            ],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "partial",
            "accessIssues": [
                {
                    "id": "adaptive_result_unavailable",
                    "scope": "Adaptive v2 relevance result",
                    "message": (
                        "Proposal inference has not been approved or run. No accepted O1, "
                        "adaptive retrieval, O1 scoring, ranking, or qrels evaluation exists."
                    ),
                }
            ],
            "datasets": {
                "baseline_metrics": baseline_metrics,
                "contract_metrics": contract_metrics,
                "proposal_metrics": proposal_metrics,
                "stage_readiness": readiness_rows,
                "stage_status": stage_rows,
                "source_receipts": source_rows,
            },
        },
        "sources": sources,
    }


def write_artifact_create_only(path: Path, artifact: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(_canonical_bytes(dict(artifact)))
    except FileExistsError as exc:
        raise FileExistsError(f"create-only report artifact already exists: {path}") from exc


def discover_portable_delivery_script() -> Path:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    candidates = list(
        (codex_home / "plugins/cache/openai-curated-remote/data-analytics").glob(
            "*/skills/build-report/scripts/deliver_portable_artifact.mjs"
        )
    )
    if not candidates:
        raise ValueError("portable report packager was not found; pass --renderer")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def deliver_html_create_only(
    *, artifact_path: Path, output_path: Path, renderer_path: Path
) -> dict[str, object]:
    """Invoke the canonical packager exactly once and publish its verified output."""

    artifact_path = Path(artifact_path)
    output_path = Path(output_path)
    renderer_path = Path(renderer_path)
    if _present(output_path):
        raise FileExistsError(f"create-only HTML report already exists: {output_path}")
    if not renderer_path.is_file():
        raise ValueError("portable report packager does not exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        completed = subprocess.run(
            [
                "node",
                str(renderer_path),
                "--input",
                str(artifact_path),
                "--output",
                str(temporary_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "TMPDIR": "/var/tmp"},
        )
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError("portable packager did not produce a non-empty HTML file")
        try:
            receipt = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("portable packager receipt is invalid") from exc
        if not isinstance(receipt, dict):
            raise RuntimeError("portable packager receipt must be an object")
        os.link(temporary_path, output_path)
        return receipt
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--proposal-preflight", type=Path, required=True)
    parser.add_argument("--baseline-rankings", type=Path, required=True)
    parser.add_argument("--v1-discovery", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer", type=Path)
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
        proposal_preflight_dir=args.proposal_preflight,
        baseline_rankings_dir=args.baseline_rankings,
        v1_discovery_dir=args.v1_discovery,
    )
    payload = build_report_payload(verified)
    artifact = build_report_artifact(payload)
    write_artifact_create_only(artifact_path, artifact)
    renderer = args.renderer or discover_portable_delivery_script()
    receipt = deliver_html_create_only(
        artifact_path=artifact_path,
        output_path=args.output,
        renderer_path=renderer,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "output": str(args.output),
                "status": payload["status"],
                "delivery": receipt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
