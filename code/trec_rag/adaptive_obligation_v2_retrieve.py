"""Plan and guard focused 1,000-hit retrieval for accepted v2 obligations.

The cache audit is deliberately transport-free.  Live execution is available only
through a separately approved, injected boundary and records exact response bytes
before decoding or normalization.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .adaptive_obligation_v2_contract import canonical_sha256
from .adaptive_obligation_v2_propose import (
    _capture_regular_file_no_symlinks,
    _fsync_directory,
    _open_directory_no_symlinks,
    _read_stable_regular_at,
)
from .adaptive_obligation_v2_validate import (
    _capture_and_verify_contract_snapshot,
    load_authenticated_validation_inventory,
)
from .det_sparse_ledger import RawTransportResponse
from .remote_client import rate_limited_session
from .remote_config import RemotePyseriniConfig
from .repo_env import find_repo_root


RETRIEVAL_HITS = 1_000
MAX_ACCEPTED_O1_PER_TOPIC = 4
MAX_RETRIEVAL_REQUESTS = 16
TIMEOUT_SECONDS = 120
TRANSPORT_RETRY_COUNT = 0
REQUEST_START_INTERVAL_SECONDS = 3
INDEX_ID = "climbmix-400b"
RETRIEVER_VERSION = "adaptive_o1_pyserini_remote_raw_first_v1"
DEFAULT_ENDPOINT = "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
LIMITER_STATE_PATH = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/"
    "pyserini_remote/rate-limit.sqlite"
)
SHARED_CACHE_DIR = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote"
)
RATE_LIMITER_IDENTITY: dict[str, object] = {
    "implementation": "requests-ratelimiter+pyrate-limiter-filelock-sqlite",
    "state_path": str(LIMITER_STATE_PATH),
    "minimum_request_start_interval_seconds": REQUEST_START_INTERVAL_SECONDS,
    "burst": 1,
    "per_host": True,
    "max_delay": None,
}
PREFLIGHT_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-preflight-v1"
JOB_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-job-v1"
CACHE_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-cache-v1"
LEDGER_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-ledger-v1"
SUMMARY_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-summary-v1"
RETRIEVAL_INPUT_SCHEMA_VERSION = "adaptive-obligation-v2-retrieval-input-v1"
RETRIEVAL_INPUT_RECEIPT_SCHEMA_VERSION = (
    "adaptive-obligation-v2-retrieval-input-receipt-v1"
)
ESTIMATED_RAW_BYTES_PER_REQUEST = 8_000_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOPIC_ORDER = {topic_id: index for index, topic_id in enumerate(PILOT_TOPIC_IDS)}
_WIRE_CONTENT_FIELDS = ("contents", "text", "body", "passage", "abstract")
_WIRE_DOCUMENT_CONTAINERS = ("doc",)


class RetrievalTransport(Protocol):
    one_shot_no_retry: bool
    request_start_interval_seconds: int
    timeout_seconds: int

    def __call__(self, job: Mapping[str, object]) -> RawTransportResponse: ...


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return " ".join(value.split())


def remove_exact_duplicate_phrases(parts: Sequence[object]) -> list[str]:
    """Remove only case-insensitive, whitespace-normalized whole-part duplicates."""

    output: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = _require_text(part, "query part")
        normalized = text.casefold()
        if normalized not in seen:
            seen.add(normalized)
            output.append(text)
    return output


def render_o1_bm25_query(
    *,
    anchor_terms: Sequence[object],
    parent_text: object,
    o1_label: object,
    narrative: object,
) -> str:
    """Render anchors + complete O0 + O1 without copying the broad narrative."""

    del narrative
    if not isinstance(anchor_terms, Sequence) or isinstance(anchor_terms, (str, bytes)):
        raise TypeError("anchor_terms must be an array of text")
    if not anchor_terms:
        raise ValueError("anchor_terms must not be empty")
    if any(not isinstance(value, str) for value in anchor_terms):
        raise TypeError("anchor_terms must contain only text")
    parts = [*anchor_terms, parent_text, o1_label]
    return " ".join(remove_exact_duplicate_phrases(parts))


def _capture_selected_files(
    path: Path,
    names: frozenset[str],
    label: str,
    *,
    exact: bool = False,
) -> dict[str, bytes]:
    try:
        descriptor = _open_directory_no_symlinks(Path(path))
        try:
            before = os.fstat(descriptor)
            available_before = set(os.listdir(descriptor))
            inventory_differs = (
                available_before != names if exact else not names <= available_before
            )
            if inventory_differs:
                raise OSError(f"{label} inventory differs")
            contents = {
                name: _read_stable_regular_at(
                    descriptor, name, require_single_link=True
                )
                for name in sorted(names)
            }
            after = os.fstat(descriptor)
            if set(os.listdir(descriptor)) != available_before or (
                before.st_dev,
                before.st_ino,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise OSError(f"{label} changed while captured")
            return contents
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc


def _json_object(source: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _jsonl_objects(source: bytes, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
        if not isinstance(value, dict) or line != _canonical_bytes(value):
            raise ValueError(f"{label}:{line_number} is not canonical")
        rows.append(value)
    return rows


def _source_chain(contract_dir: Path, expected_contract_sha256: str) -> dict[str, object]:
    contract, contract_sha256 = _capture_and_verify_contract_snapshot(
        Path(contract_dir), expected_receipt_sha256=expected_contract_sha256
    )
    receipt = contract.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("retrieval source contract receipt is missing")
    bindings = receipt.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("retrieval source contract bindings are missing")
    base_dir_text = bindings.get("contract_dir")
    summary_sha256 = bindings.get("contract_summary_sha256")
    if (
        not isinstance(base_dir_text, str)
        or not Path(base_dir_text).is_absolute()
        or not isinstance(summary_sha256, str)
        or not _SHA256_RE.fullmatch(summary_sha256)
    ):
        raise ValueError("retrieval adaptive contract binding differs")
    base_dir = Path(base_dir_text)
    base_contents = _capture_selected_files(
        base_dir,
        frozenset({"summary.json", "manifest.json", "obligations.jsonl"}),
        "adaptive evidence contract",
    )
    if _sha256(base_contents["summary.json"]) != summary_sha256:
        raise ValueError("retrieval adaptive contract summary hash differs")
    summary = _json_object(base_contents["summary.json"], "adaptive contract summary")
    base_manifest = _json_object(
        base_contents["manifest.json"], "adaptive contract manifest"
    )
    artifact_sha = summary.get("artifact_sha256")
    if (
        summary.get("status") != "complete"
        or summary.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or not isinstance(artifact_sha, Mapping)
        or artifact_sha.get("manifest.json") != _sha256(base_contents["manifest.json"])
        or artifact_sha.get("obligations.jsonl")
        != _sha256(base_contents["obligations.jsonl"])
    ):
        raise ValueError("retrieval adaptive contract artifact binding differs")
    sources = base_manifest.get("sources")
    manifest_binding = sources.get("manifest") if isinstance(sources, Mapping) else None
    if (
        not isinstance(manifest_binding, Mapping)
        or not isinstance(manifest_binding.get("path"), str)
        or Path(str(manifest_binding["path"])).is_absolute()
        or not isinstance(manifest_binding.get("sha256"), str)
        or not _SHA256_RE.fullmatch(str(manifest_binding["sha256"]))
    ):
        raise ValueError("retrieval source manifest binding differs")
    repo_root = find_repo_root(base_dir)
    manifest_path = repo_root / str(manifest_binding["path"])
    try:
        manifest_source = _capture_regular_file_no_symlinks(manifest_path)
    except OSError as exc:
        raise ValueError("retrieval source manifest is missing or unsafe") from exc
    if _sha256(manifest_source) != manifest_binding["sha256"]:
        raise ValueError("retrieval source manifest hash differs")
    source_manifest = _json_object(manifest_source, "retrieval source manifest")
    if (
        source_manifest.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or source_manifest.get("qrels_opened") is not False
    ):
        raise ValueError("retrieval source manifest topic boundary differs")
    obligations = _jsonl_objects(
        base_contents["obligations.jsonl"], "adaptive obligations"
    )
    return {
        "contract": contract,
        "contract_receipt_sha256": contract_sha256,
        "adaptive_summary_sha256": summary_sha256,
        "adaptive_manifest_sha256": _sha256(base_contents["manifest.json"]),
        "adaptive_obligations_sha256": _sha256(base_contents["obligations.jsonl"]),
        "obligations": obligations,
        "source_manifest": source_manifest,
        "source_manifest_path": str(manifest_binding["path"]),
        "source_manifest_sha256": str(manifest_binding["sha256"]),
    }


def _retrieval_input_rows(
    accepted: Sequence[Mapping[str, object]],
    *,
    validation_receipt_sha256: str,
    chain: Mapping[str, object],
) -> list[dict[str, object]]:
    contract = chain.get("contract")
    obligations = chain.get("obligations")
    source_manifest = chain.get("source_manifest")
    if (
        not isinstance(contract, Mapping)
        or not isinstance(obligations, list)
        or not isinstance(source_manifest, Mapping)
    ):
        raise ValueError("retrieval authenticated source chain differs")
    parents = contract.get("parents")
    if not isinstance(parents, list):
        raise ValueError("retrieval source parents are missing")
    parent_by_id = {str(row.get("parent_id")): row for row in parents if isinstance(row, Mapping)}
    obligation_by_id = {
        str(row.get("obligation_id")): row
        for row in obligations
        if isinstance(row, Mapping)
    }
    broad_by_topic = {
        str(row.get("topic_id")): row
        for row in obligations
        if isinstance(row, Mapping) and row.get("kind") == "broad"
    }
    topics = source_manifest.get("topics")
    facets = source_manifest.get("facets")
    if not isinstance(topics, list) or not isinstance(facets, list):
        raise ValueError("retrieval source manifest rows are missing")
    topic_by_id = {str(row.get("topic_id")): row for row in topics if isinstance(row, Mapping)}
    facet_by_id = {str(row.get("facet_id")): row for row in facets if isinstance(row, Mapping)}
    rows: list[dict[str, object]] = []
    for accepted_row in accepted:
        topic_id = str(accepted_row.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in _TOPIC_ORDER:
            raise ValueError("retrieval accepted topic is unknown")
        parent_id = str(accepted_row.get("parent_id"))
        parent = parent_by_id.get(parent_id)
        obligation = obligation_by_id.get(parent_id)
        facet = facet_by_id.get(parent_id)
        broad = broad_by_topic.get(topic_id)
        topic = topic_by_id.get(topic_id)
        if not all(isinstance(value, Mapping) for value in (parent, obligation, facet, broad, topic)):
            raise ValueError("retrieval accepted parent source join is incomplete")
        assert isinstance(parent, Mapping)
        assert isinstance(obligation, Mapping)
        assert isinstance(facet, Mapping)
        assert isinstance(broad, Mapping)
        assert isinstance(topic, Mapping)
        parent_text = parent.get("text")
        parent_query = parent.get("query")
        narrative = topic.get("query")
        anchors = facet.get("anchor_terms")
        label = accepted_row.get("label")
        if (
            not isinstance(parent_text, str)
            or parent.get("text_sha256") != _sha256(parent_text.encode("utf-8"))
            or not isinstance(parent_query, str)
            or parent.get("query_sha256") != _sha256(parent_query.encode("utf-8"))
            or not isinstance(narrative, str)
            or not isinstance(label, str)
            or not label.strip()
            or not isinstance(anchors, list)
            or not anchors
            or any(not isinstance(anchor, str) or not anchor.strip() for anchor in anchors)
            or parent.get("topic_id") != topic_id
            or obligation.get("topic_id") != topic_id
            or obligation.get("text") != parent_text
            or obligation.get("query") != parent_query
            or obligation.get("manifest_order") != parent.get("manifest_order")
            or obligation.get("source_facet_id") != parent_id
            or obligation.get("anchor_terms") != anchors
            or facet.get("topic_id") != topic_id
            or facet.get("obligation") != parent_text
            or broad.get("text") != narrative
            or broad.get("query") != narrative
            or parent_query != f"{narrative}\n\nExplicit obligation:\n{parent_text}"
        ):
            raise ValueError("retrieval O0/narrative/anchor source join differs")
        support_names = (
            "proposal_document_ids",
            "validation_support_unit_ids",
            "validation_document_ids",
            "support_document_ids",
        )
        if any(
            not isinstance(accepted_row.get(name), list)
            or any(not isinstance(value, str) or not value for value in accepted_row[name])
            for name in support_names
        ):
            raise ValueError("retrieval accepted support lineage differs")
        identity = {
            "accepted_validation_sha256": canonical_sha256(dict(accepted_row)),
            "validation_receipt_sha256": validation_receipt_sha256,
            "topic_id": topic_id,
            "parent_id": parent_id,
            "proposal_id": accepted_row.get("proposal_id"),
            "validation_job_id": accepted_row.get("validation_job_id"),
            "o1_label": label,
        }
        row: dict[str, object] = {
            "schema_version": RETRIEVAL_INPUT_SCHEMA_VERSION,
            "input_id": canonical_sha256(identity),
            **identity,
            "parent_manifest_order": parent["manifest_order"],
            "anchor_terms": list(anchors),
            "parent_text": parent_text,
            "parent_text_sha256": parent["text_sha256"],
            "parent_query": parent_query,
            "parent_query_sha256": parent["query_sha256"],
            "narrative": narrative,
            "narrative_sha256": _sha256(narrative.encode("utf-8")),
            "o1_label_sha256": _sha256(label.encode("utf-8")),
            "proposal_fold": accepted_row.get("proposal_fold"),
            "validation_fold": accepted_row.get("validation_fold"),
            "proposal_document_ids": list(accepted_row["proposal_document_ids"]),
            "validation_support_unit_ids": list(
                accepted_row["validation_support_unit_ids"]
            ),
            "validation_document_ids": list(accepted_row["validation_document_ids"]),
            "support_document_ids": list(accepted_row["support_document_ids"]),
            "validation_result_sha256": accepted_row.get(
                "validation_result_sha256"
            ),
            "source_manifest": {
                "path": chain["source_manifest_path"],
                "sha256": chain["source_manifest_sha256"],
                "facet_id": parent_id,
            },
            "source_chain": {
                "v2_contract_receipt_sha256": chain["contract_receipt_sha256"],
                "adaptive_contract_summary_sha256": chain[
                    "adaptive_summary_sha256"
                ],
                "adaptive_contract_manifest_sha256": chain[
                    "adaptive_manifest_sha256"
                ],
                "adaptive_contract_obligations_sha256": chain[
                    "adaptive_obligations_sha256"
                ],
            },
        }
        rows.append(row)
    return rows


def _publish_retrieval_inputs(destination: Path, contents: Mapping[str, bytes]) -> None:
    destination = Path(destination)
    if not destination.name or destination.name in {".", ".."}:
        raise ValueError("retrieval input output path is unsafe")
    try:
        parent_fd = _open_directory_no_symlinks(destination.parent)
    except OSError as exc:
        raise ValueError("retrieval input output parent is missing or unsafe") from exc
    staging_name = f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging_fd: int | None = None
    published = False
    try:
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_fd = _open_dir_at(parent_fd, staging_name, create=False)
        for name, source in contents.items():
            _write_exclusive_at(staging_fd, name, source)
        os.fsync(staging_fd)
        _rename_noreplace_at(parent_fd, staging_name, destination.name)
        published = True
        os.fsync(parent_fd)
    finally:
        if staging_fd is not None:
            if not published:
                for name in contents:
                    try:
                        os.unlink(name, dir_fd=staging_fd)
                    except FileNotFoundError:
                        pass
            os.close(staging_fd)
        if not published:
            try:
                os.rmdir(staging_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _retrieval_input_material(
    *,
    validation_output_dir: Path,
    validation_preflight_dir: Path,
    validation_approval_path: Path,
    validation_ledger_dir: Path,
    proposal_inventory_dir: Path,
    proposal_preflight_dir: Path,
    proposal_ledger_dir: Path,
    contract_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, bytes]]:
    authenticated = load_authenticated_validation_inventory(
        output_dir=validation_output_dir,
        preflight_dir=validation_preflight_dir,
        approval_path=validation_approval_path,
        ledger_dir=validation_ledger_dir,
        proposal_inventory_dir=proposal_inventory_dir,
        proposal_preflight_dir=proposal_preflight_dir,
        proposal_ledger_dir=proposal_ledger_dir,
        source_contract_dir=contract_dir,
    )
    accepted = authenticated.get("accepted")
    validation_receipt = authenticated.get("receipt")
    validation_receipt_sha256 = authenticated.get("receipt_sha256")
    if (
        not isinstance(accepted, list)
        or any(not isinstance(row, Mapping) for row in accepted)
        or not isinstance(validation_receipt, Mapping)
        or not isinstance(validation_receipt_sha256, str)
        or not _SHA256_RE.fullmatch(validation_receipt_sha256)
    ):
        raise ValueError("retrieval requires authenticated Task 4 acceptance")
    expected_contract_sha = validation_receipt.get("source_contract_receipt_sha256")
    if not isinstance(expected_contract_sha, str) or not _SHA256_RE.fullmatch(expected_contract_sha):
        raise ValueError("retrieval Task 4 source contract binding differs")
    chain = _source_chain(contract_dir, expected_contract_sha)
    rows = _retrieval_input_rows(
        accepted,
        validation_receipt_sha256=validation_receipt_sha256,
        chain=chain,
    )
    input_source = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    receipt: dict[str, object] = {
        "schema_version": RETRIEVAL_INPUT_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "input_count": len(rows),
        "inputs_sha256": _sha256(input_source),
        "validation_inventory": {
            "output_dir": str(validation_output_dir.absolute()),
            "preflight_dir": str(validation_preflight_dir.absolute()),
            "approval_path": str(validation_approval_path.absolute()),
            "ledger_dir": str(validation_ledger_dir.absolute()),
            "proposal_inventory_dir": str(proposal_inventory_dir.absolute()),
            "proposal_preflight_dir": str(proposal_preflight_dir.absolute()),
            "proposal_ledger_dir": str(proposal_ledger_dir.absolute()),
            "receipt_sha256": validation_receipt_sha256,
            "preflight_sha256": validation_receipt[
                "validation_preflight_sha256"
            ],
            "run_anchor_sha256": validation_receipt["run_anchor_sha256"],
            "completion_sha256": validation_receipt["completion_sha256"],
            "proposal_receipt_sha256": validation_receipt[
                "proposal_receipt_sha256"
            ],
        },
        "source_contract": {
            "path": str(Path(contract_dir).absolute()),
            "receipt_sha256": chain["contract_receipt_sha256"],
            "adaptive_summary_sha256": chain["adaptive_summary_sha256"],
            "adaptive_manifest_sha256": chain["adaptive_manifest_sha256"],
            "adaptive_obligations_sha256": chain[
                "adaptive_obligations_sha256"
            ],
            "source_manifest_path": chain["source_manifest_path"],
            "source_manifest_sha256": chain["source_manifest_sha256"],
        },
        "artifacts": {
            "inputs.jsonl": {
                "rows": len(rows),
                "bytes": len(input_source),
                "sha256": _sha256(input_source),
            }
        },
    }
    return rows, receipt, {
        "inputs.jsonl": input_source,
        "receipt.json": _pretty_bytes(receipt),
    }


def freeze_retrieval_inputs(
    *,
    validation_output_dir: Path,
    validation_preflight_dir: Path,
    validation_approval_path: Path,
    validation_ledger_dir: Path,
    proposal_inventory_dir: Path,
    proposal_preflight_dir: Path,
    proposal_ledger_dir: Path,
    contract_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Freeze Task 5 inputs from the replay-authenticated Task 4 product."""

    _rows, receipt, contents = _retrieval_input_material(
        validation_output_dir=Path(validation_output_dir),
        validation_preflight_dir=Path(validation_preflight_dir),
        validation_approval_path=Path(validation_approval_path),
        validation_ledger_dir=Path(validation_ledger_dir),
        proposal_inventory_dir=Path(proposal_inventory_dir),
        proposal_preflight_dir=Path(proposal_preflight_dir),
        proposal_ledger_dir=Path(proposal_ledger_dir),
        contract_dir=Path(contract_dir),
    )
    _publish_retrieval_inputs(Path(output_dir), contents)
    return receipt


def load_authenticated_retrieval_inputs(input_dir: Path) -> dict[str, object]:
    contents = _capture_selected_files(
        Path(input_dir),
        frozenset({"inputs.jsonl", "receipt.json"}),
        "retrieval input inventory",
        exact=True,
    )
    receipt = _json_object(contents["receipt.json"], "retrieval input receipt")
    if (
        contents["receipt.json"] != _pretty_bytes(receipt)
        or receipt.get("schema_version") != RETRIEVAL_INPUT_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
    ):
        raise ValueError("retrieval input receipt differs")
    validation = receipt.get("validation_inventory")
    source = receipt.get("source_contract")
    if not isinstance(validation, Mapping) or not isinstance(source, Mapping):
        raise ValueError("retrieval input source bindings are missing")
    path_names = (
        "output_dir",
        "preflight_dir",
        "approval_path",
        "ledger_dir",
        "proposal_inventory_dir",
        "proposal_preflight_dir",
        "proposal_ledger_dir",
    )
    if any(
        not isinstance(validation.get(name), str)
        or not Path(str(validation[name])).is_absolute()
        for name in path_names
    ) or not isinstance(source.get("path"), str) or not Path(str(source["path"])).is_absolute():
        raise ValueError("retrieval input source paths differ")
    rows, expected_receipt, expected_contents = _retrieval_input_material(
        validation_output_dir=Path(str(validation["output_dir"])),
        validation_preflight_dir=Path(str(validation["preflight_dir"])),
        validation_approval_path=Path(str(validation["approval_path"])),
        validation_ledger_dir=Path(str(validation["ledger_dir"])),
        proposal_inventory_dir=Path(str(validation["proposal_inventory_dir"])),
        proposal_preflight_dir=Path(str(validation["proposal_preflight_dir"])),
        proposal_ledger_dir=Path(str(validation["proposal_ledger_dir"])),
        contract_dir=Path(str(source["path"])),
    )
    if contents != expected_contents or receipt != expected_receipt:
        raise ValueError("retrieval inputs differ from authenticated source replay")
    observed_rows = _jsonl_objects(contents["inputs.jsonl"], "inputs.jsonl")
    if observed_rows != rows:
        raise ValueError("retrieval inputs differ from authenticated source replay")
    return {"inputs": rows, "receipt": receipt, "receipt_sha256": _sha256(contents["receipt.json"])}


def _protected_precheck(rows: object) -> list[Mapping[str, object]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("accepted O1 rows must be an array")
    materialized: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("accepted O1 rows must contain objects")
        topic_id = str(row.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        materialized.append(row)
    if len(materialized) > MAX_RETRIEVAL_REQUESTS:
        raise ValueError(f"retrieval permits at most {MAX_RETRIEVAL_REQUESTS} requests")
    return materialized


def _validated_endpoint(value: object) -> str:
    endpoint = _require_text(value, "endpoint")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain a query or fragment")
    return endpoint


def _build_retrieval_jobs_from_inputs(
    accepted_o1_rows: Sequence[Mapping[str, object]],
    *,
    endpoint: object = DEFAULT_ENDPOINT,
    index_id: object = INDEX_ID,
) -> list[dict[str, object]]:
    """Freeze one exact 1,000-hit request for every accepted O1."""

    rows = _protected_precheck(accepted_o1_rows)
    endpoint_text = _validated_endpoint(endpoint)
    index_text = _require_text(index_id, "index_id")
    topic_counts = Counter(str(row.get("topic_id")) for row in rows)
    if any(count > MAX_ACCEPTED_O1_PER_TOPIC for count in topic_counts.values()):
        raise ValueError("retrieval permits at most four accepted O1 records per topic")

    validated: list[tuple[tuple[object, ...], Mapping[str, object]]] = []
    seen_ids: set[str] = set()
    for row in rows:
        topic_id = str(row.get("topic_id"))
        if topic_id not in _TOPIC_ORDER:
            raise ValueError("accepted O1 topic is unknown")
        frozen_input = row.get("schema_version") == RETRIEVAL_INPUT_SCHEMA_VERSION
        if not frozen_input and (
            row.get("accepted") is not True or row.get("decision") != "SUPPORTED"
        ):
            raise ValueError("retrieval requires an accepted SUPPORTED O1")
        proposal_id = _require_text(row.get("proposal_id"), "accepted O1 ID")
        if proposal_id in seen_ids:
            raise ValueError("accepted O1 IDs must be unique")
        seen_ids.add(proposal_id)
        parent_id = _require_text(row.get("parent_id"), "parent_id")
        manifest_order = row.get("parent_manifest_order")
        if type(manifest_order) is not int or manifest_order < 0:
            raise ValueError("parent_manifest_order must be a non-negative integer")
        _require_text(row.get("parent_text"), "parent_text")
        label_name = "o1_label" if frozen_input else "label"
        _require_text(row.get(label_name), "label")
        anchors = row.get("anchor_terms")
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
            raise TypeError("anchor_terms must be an array of text")
        if not anchors or any(not isinstance(value, str) for value in anchors):
            raise ValueError("anchor_terms must contain non-empty text")
        validated.append(
            (
                (
                    _TOPIC_ORDER[topic_id],
                    manifest_order,
                    parent_id.casefold(),
                    proposal_id.casefold(),
                ),
                row,
            )
        )

    jobs: list[dict[str, object]] = []
    for _key, row in sorted(validated, key=lambda pair: pair[0]):
        query_text = render_o1_bm25_query(
            anchor_terms=row["anchor_terms"],  # type: ignore[arg-type]
            parent_text=row["parent_text"],
            o1_label=row["o1_label"]
            if row.get("schema_version") == RETRIEVAL_INPUT_SCHEMA_VERSION
            else row["label"],
            narrative=row.get("narrative"),
        )
        query_sha256 = _sha256(query_text.encode("utf-8"))
        accepted_o1_sha256 = canonical_sha256(dict(row))
        identity: dict[str, object] = {
            "topic_id": str(row["topic_id"]),
            "parent_id": str(row["parent_id"]),
            "accepted_o1_id": str(row["proposal_id"]),
            "accepted_o1_sha256": accepted_o1_sha256,
            "query_text": query_text,
            "query_sha256": query_sha256,
            "endpoint": endpoint_text,
            "index_id": index_text,
            "retriever_version": RETRIEVER_VERSION,
            "hits": RETRIEVAL_HITS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "transport_retry_count": TRANSPORT_RETRY_COUNT,
            "rate_limiter": dict(RATE_LIMITER_IDENTITY),
        }
        request_key = canonical_sha256(identity)
        jobs.append(
            {
                "schema_version": JOB_SCHEMA_VERSION,
                "job_id": request_key,
                "request_key": request_key,
                "topic_id": identity["topic_id"],
                "parent_id": identity["parent_id"],
                "accepted_o1_id": identity["accepted_o1_id"],
                "accepted_o1_sha256": accepted_o1_sha256,
                "query_text": query_text,
                "query_sha256": query_sha256,
                "endpoint": endpoint_text,
                "index_id": index_text,
                "retriever_version": RETRIEVER_VERSION,
                "hits": RETRIEVAL_HITS,
                "timeout_seconds": TIMEOUT_SECONDS,
                "transport_retry_count": TRANSPORT_RETRY_COUNT,
                "rate_limiter": dict(RATE_LIMITER_IDENTITY),
                "request_identity": identity,
                "retrieval_input": dict(row),
            }
        )
    return jobs


def build_retrieval_jobs(
    retrieval_input_dir: Path,
    *,
    endpoint: object = DEFAULT_ENDPOINT,
    index_id: object = INDEX_ID,
) -> list[dict[str, object]]:
    """Build jobs only from replay-authenticated, source-joined Task 5 inputs."""

    authenticated = load_authenticated_retrieval_inputs(Path(retrieval_input_dir))
    inputs = authenticated.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("retrieval input inventory differs")
    return _build_retrieval_jobs_from_inputs(
        inputs, endpoint=endpoint, index_id=index_id
    )


def _normalize_content_text(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _extract_document_container(value: object) -> str:
    """Read only declared content fields from a hosted document container."""

    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return ""
        if isinstance(decoded, (Mapping, list)):
            return _extract_document_container(decoded)
        return ""
    if isinstance(value, Mapping):
        for field in _WIRE_CONTENT_FIELDS:
            text = _normalize_content_text(value.get(field))
            if text:
                return text
        for container in _WIRE_DOCUMENT_CONTAINERS:
            if container in value:
                text = _extract_document_container(value[container])
                if text:
                    return text
        return ""
    if isinstance(value, list):
        parts = [
            text
            for item in value
            if isinstance(item, Mapping)
            for text in (_extract_document_container(item),)
            if text
        ]
        return " ".join(parts)
    return ""


def _extract_candidate_content(candidate: Mapping[str, object]) -> str:
    for container in _WIRE_DOCUMENT_CONTAINERS:
        if container in candidate:
            text = _extract_document_container(candidate[container])
            if text:
                return text
    for field in _WIRE_CONTENT_FIELDS:
        text = _normalize_content_text(candidate.get(field))
        if text:
            return text
    return ""


def _normalize_response(raw: bytes) -> tuple[dict[str, object], ...]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("response is not valid UTF-8 JSON") from exc
    candidates = None
    if isinstance(payload, Mapping):
        for key in ("candidates", "hits", "results"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                candidates = value
                break
    if not isinstance(candidates, list) or len(candidates) != RETRIEVAL_HITS:
        raise ValueError("response must contain exactly 1000 unique text-bearing candidates")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for position, value in enumerate(candidates, start=1):
        if not isinstance(value, Mapping):
            raise ValueError("response must contain exactly 1000 unique text-bearing candidates")
        docid = value.get("docid") or value.get("id") or value.get("_id")
        rank = value.get("rank", position)
        score = value.get("score", 0.0)
        text = _extract_candidate_content(value)
        if (
            not isinstance(docid, str)
            or not docid.strip()
            or docid.strip() in seen
            or type(rank) is not int
            or rank != position
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not text
        ):
            raise ValueError("response must contain exactly 1000 unique text-bearing candidates")
        docid = docid.strip()
        seen.add(docid)
        normalized.append(
            {
                "docid": docid,
                "rank": position,
                "score": float(score),
                "text": text,
                "text_sha256": _sha256(text.encode("utf-8")),
            }
        )
    return tuple(normalized)


def _verified_cache_sources(
    job: Mapping[str, object],
    raw: bytes,
    candidate_source: bytes,
    manifest_source: bytes,
) -> dict[str, object]:
    request_key = str(job.get("request_key"))
    try:
        candidates = json.loads(candidate_source)
        manifest = json.loads(manifest_source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("exact cache JSON is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest_source != _pretty_bytes(manifest)
        or candidate_source != _pretty_bytes(candidates)
        or manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("request_key") != request_key
        or manifest.get("request_identity") != job.get("request_identity")
        or manifest.get("query_text") != job.get("query_text")
        or manifest.get("raw_sha256") != _sha256(raw)
        or manifest.get("candidates_sha256") != _sha256(candidate_source)
        or manifest.get("candidate_count") != RETRIEVAL_HITS
        or not isinstance(candidates, list)
    ):
        raise ValueError("exact cache identity or content hash mismatch")
    normalized = _normalize_response(raw)
    if list(normalized) != candidates:
        raise ValueError("exact cache normalized candidates mismatch")
    return {
        "raw": raw,
        "candidates": normalized,
        "hit": True,
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "candidate_count": RETRIEVAL_HITS,
    }


class _SecureCacheStore:
    """Hold the approved cache root descriptor through audit and publication."""

    def __init__(self, root: Path) -> None:
        try:
            self.root_fd = _open_directory_no_symlinks(Path(root))
        except OSError as exc:
            raise ValueError("approved cache root is missing or unsafe") from exc

    def _prefix(self, request_key: str, *, create: bool) -> tuple[int, int]:
        adaptive = _open_dir_at(self.root_fd, "adaptive-obligation-v2", create=create)
        try:
            prefix = _open_dir_at(adaptive, request_key[:2], create=create)
        except BaseException:
            os.close(adaptive)
            raise
        return adaptive, prefix

    def load(self, job: Mapping[str, object]) -> dict[str, object] | None:
        key = str(job["request_key"])
        try:
            adaptive, prefix = self._prefix(key, create=False)
        except FileNotFoundError:
            return None
        try:
            names = (
                f"{key}.raw.json",
                f"{key}.candidates.json",
                f"{key}.manifest.json",
            )
            exists: list[bool] = []
            for name in names:
                try:
                    os.stat(name, dir_fd=prefix, follow_symlinks=False)
                    exists.append(True)
                except FileNotFoundError:
                    exists.append(False)
            if not any(exists):
                return None
            if not all(exists):
                raise ValueError(f"partial exact cache exists for {key}")
            raw = _read_stable_regular_at(prefix, names[0], require_single_link=True)
            candidates = _read_stable_regular_at(
                prefix, names[1], require_single_link=True
            )
            manifest = _read_stable_regular_at(
                prefix, names[2], require_single_link=True
            )
            return _verified_cache_sources(job, raw, candidates, manifest)
        finally:
            os.close(prefix)
            os.close(adaptive)

    def store(
        self,
        job: Mapping[str, object],
        raw: bytes,
        candidates: Sequence[Mapping[str, object]],
    ) -> None:
        key = str(job["request_key"])
        adaptive, prefix = self._prefix(key, create=True)
        try:
            candidate_source = _pretty_bytes(list(candidates))
            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "request_key": key,
                "request_identity": job["request_identity"],
                "query_text": job["query_text"],
                "raw_sha256": _sha256(raw),
                "candidates_sha256": _sha256(candidate_source),
                "candidate_count": RETRIEVAL_HITS,
            }
            _write_exclusive_at(prefix, f"{key}.raw.json", raw)
            _write_exclusive_at(
                prefix, f"{key}.candidates.json", candidate_source
            )
            _write_exclusive_at(
                prefix, f"{key}.manifest.json", _pretty_bytes(manifest)
            )
        finally:
            os.close(prefix)
            os.close(adaptive)

    def close(self) -> None:
        os.fsync(self.root_fd)
        os.close(self.root_fd)


def _cache_status(value: object, job: Mapping[str, object]) -> dict[str, object]:
    if value is None:
        return {"request_key": job["request_key"], "hit": False, "raw_bytes": 0}
    if not isinstance(value, Mapping):
        raise ValueError("cache probe must return an object or None")
    allowed_probe_fields = {
        "request_key",
        "hit",
        "raw_bytes",
        "raw_sha256",
        "candidate_count",
        "raw",
        "candidates",
    }
    if not set(value) <= allowed_probe_fields:
        raise ValueError("cache probe fields differ")
    if "request_key" in value and value.get("request_key") != job["request_key"]:
        raise ValueError("cache probe request_key differs")
    hit = value.get("hit")
    raw_bytes = value.get("raw_bytes")
    candidate_count = value.get("candidate_count")
    raw_sha256 = value.get("raw_sha256")
    if (
        hit is not True
        or type(raw_bytes) is not int
        or raw_bytes < 0
        or candidate_count != RETRIEVAL_HITS
        or not isinstance(raw_sha256, str)
        or not _SHA256_RE.fullmatch(raw_sha256)
    ):
        raise ValueError("cache probe returned an invalid verified-hit record")
    return {
        "request_key": job["request_key"],
        "hit": True,
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_sha256,
        "candidate_count": RETRIEVAL_HITS,
    }


def _verify_job(job: object) -> Mapping[str, object]:
    if not isinstance(job, Mapping):
        raise ValueError("retrieval job must be an object")
    job_fields = {
        "schema_version",
        "job_id",
        "request_key",
        "topic_id",
        "parent_id",
        "accepted_o1_id",
        "accepted_o1_sha256",
        "query_text",
        "query_sha256",
        "endpoint",
        "index_id",
        "retriever_version",
        "hits",
        "timeout_seconds",
        "transport_retry_count",
        "rate_limiter",
        "request_identity",
        "retrieval_input",
    }
    if set(job) != job_fields:
        raise ValueError("retrieval job fields differ")
    identity = job.get("request_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("retrieval request identity is missing")
    identity_fields = {
        "topic_id",
        "parent_id",
        "accepted_o1_id",
        "accepted_o1_sha256",
        "query_text",
        "query_sha256",
        "endpoint",
        "index_id",
        "retriever_version",
        "hits",
        "timeout_seconds",
        "transport_retry_count",
        "rate_limiter",
    }
    if set(identity) != identity_fields:
        raise ValueError("retrieval request identity fields differ")
    limiter = identity.get("rate_limiter")
    if not isinstance(limiter, Mapping) or set(limiter) != set(RATE_LIMITER_IDENTITY):
        raise ValueError("retrieval rate limiter identity fields differ")
    retrieval_input = job.get("retrieval_input")
    if (
        not isinstance(retrieval_input, Mapping)
        or canonical_sha256(dict(retrieval_input))
        != identity.get("accepted_o1_sha256")
    ):
        raise ValueError("retrieval frozen input differs")
    request_key = job.get("request_key")
    if (
        job.get("schema_version") != JOB_SCHEMA_VERSION
        or not isinstance(request_key, str)
        or request_key != canonical_sha256(dict(identity))
        or job.get("job_id") != request_key
        or job.get("query_text") != identity.get("query_text")
        or job.get("query_sha256") != _sha256(str(job.get("query_text")).encode("utf-8"))
        or job.get("query_sha256") != identity.get("query_sha256")
    ):
        raise ValueError("retrieval request identity differs")
    topic_id = str(identity.get("topic_id"))
    if topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")
    expected = {
        "topic_id": job.get("topic_id"),
        "parent_id": job.get("parent_id"),
        "accepted_o1_id": job.get("accepted_o1_id"),
        "accepted_o1_sha256": job.get("accepted_o1_sha256"),
        "query_text": job.get("query_text"),
        "query_sha256": job.get("query_sha256"),
        "endpoint": job.get("endpoint"),
        "index_id": job.get("index_id"),
        "retriever_version": RETRIEVER_VERSION,
        "hits": RETRIEVAL_HITS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "transport_retry_count": TRANSPORT_RETRY_COUNT,
        "rate_limiter": RATE_LIMITER_IDENTITY,
    }
    if (
        dict(identity) != expected
        or topic_id not in _TOPIC_ORDER
        or any(
            not isinstance(identity.get(name), str) or not str(identity[name]).strip()
            for name in (
                "topic_id",
                "parent_id",
                "accepted_o1_id",
                "query_text",
                "endpoint",
                "index_id",
                "retriever_version",
            )
        )
        or any(
            not isinstance(identity.get(name), str)
            or not _SHA256_RE.fullmatch(str(identity[name]))
            for name in ("accepted_o1_sha256", "query_sha256")
        )
        or _validated_endpoint(identity["endpoint"]) != identity["endpoint"]
    ):
        raise ValueError("retrieval request identity differs")
    return job


def _verify_receipt(receipt: object) -> dict[str, object]:
    if not isinstance(receipt, dict):
        raise ValueError("retrieval preflight receipt must be an object")
    expected_fields = {
        "schema_version",
        "status",
        "topic_ids",
        "retrieval_input_dir",
        "retrieval_input_receipt_sha256",
        "planned_request_count",
        "verified_cache_hits",
        "verified_cache_misses",
        "expected_raw_rows",
        "observed_cached_raw_bytes",
        "estimated_new_raw_bytes",
        "estimated_total_raw_bytes",
        "primary_external_attempts",
        "maximum_external_attempts",
        "hits",
        "timeout_seconds",
        "transport_retry_count",
        "rate_limiter",
        "endpoint",
        "index_id",
        "cache_root",
        "output_dir",
        "cache_audit",
        "request_inventory_sha256",
        "requests",
        "qrels_opened",
        "network_call_count",
        "retrieval_call_count",
        "paid_call_count",
    }
    if set(receipt) != expected_fields:
        raise ValueError("retrieval preflight fields differ")
    if receipt.get("topic_ids") != list(PILOT_TOPIC_IDS):
        raise ValueError("retrieval preflight topic_ids differ")
    jobs = receipt.get("requests")
    if not isinstance(jobs, list):
        raise ValueError("retrieval preflight request inventory is missing")
    verified_jobs = [_verify_job(job) for job in jobs]
    if len(jobs) > MAX_RETRIEVAL_REQUESTS:
        raise ValueError("retrieval preflight exceeds the 16-request ceiling")
    counts = Counter(str(job["topic_id"]) for job in verified_jobs)
    if any(count > MAX_ACCEPTED_O1_PER_TOPIC for count in counts.values()):
        raise ValueError("retrieval preflight exceeds the per-topic ceiling")
    keys = [str(job["request_key"]) for job in verified_jobs]
    cache_audit = receipt.get("cache_audit")
    if (
        len(set(keys)) != len(keys)
        or not isinstance(cache_audit, list)
        or len(cache_audit) != len(jobs)
    ):
        raise ValueError("retrieval preflight inventories differ")
    for job, status in zip(verified_jobs, cache_audit, strict=True):
        if not isinstance(status, Mapping) or status.get("request_key") != job["request_key"]:
            raise ValueError("retrieval preflight cache audit differs")
        expected_status_fields = (
            {"request_key", "hit", "raw_bytes", "raw_sha256", "candidate_count"}
            if status.get("hit") is True
            else {"request_key", "hit", "raw_bytes"}
        )
        if set(status) != expected_status_fields or type(status.get("hit")) is not bool:
            raise ValueError("retrieval preflight cache audit fields differ")
        _cache_status(status if status.get("hit") is True else None, job)
    hits = sum(status.get("hit") is True for status in cache_audit if isinstance(status, Mapping))
    misses = len(jobs) - hits
    cached_bytes = sum(
        int(status.get("raw_bytes", 0))
        for status in cache_audit
        if isinstance(status, Mapping) and status.get("hit") is True
    )
    expected_static = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "complete",
        "planned_request_count": len(jobs),
        "verified_cache_hits": hits,
        "verified_cache_misses": misses,
        "expected_raw_rows": len(jobs) * RETRIEVAL_HITS,
        "observed_cached_raw_bytes": cached_bytes,
        "estimated_new_raw_bytes": misses * ESTIMATED_RAW_BYTES_PER_REQUEST,
        "estimated_total_raw_bytes": cached_bytes + misses * ESTIMATED_RAW_BYTES_PER_REQUEST,
        "primary_external_attempts": misses,
        "maximum_external_attempts": misses,
        "hits": RETRIEVAL_HITS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "transport_retry_count": TRANSPORT_RETRY_COUNT,
        "rate_limiter": RATE_LIMITER_IDENTITY,
        "endpoint": (
            str(verified_jobs[0]["endpoint"])
            if verified_jobs
            else receipt.get("endpoint")
        ),
        "index_id": (
            str(verified_jobs[0]["index_id"])
            if verified_jobs
            else receipt.get("index_id")
        ),
        "request_inventory_sha256": canonical_sha256(jobs),
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "paid_call_count": 0,
    }
    for name, expected in expected_static.items():
        if receipt.get(name) != expected:
            raise ValueError(f"retrieval preflight {name} differs")
    cache_root = receipt.get("cache_root")
    output_dir = receipt.get("output_dir")
    retrieval_input_dir = receipt.get("retrieval_input_dir")
    retrieval_input_receipt_sha256 = receipt.get(
        "retrieval_input_receipt_sha256"
    )
    if (
        receipt.get("endpoint") != _validated_endpoint(receipt.get("endpoint"))
        or not isinstance(receipt.get("index_id"), str)
        or not str(receipt["index_id"]).strip()
        or any(
            job.get("endpoint") != receipt["endpoint"]
            or job.get("index_id") != receipt["index_id"]
            for job in verified_jobs
        )
        or not isinstance(cache_root, str)
        or not Path(cache_root).is_absolute()
        or not isinstance(output_dir, str)
        or not Path(output_dir).is_absolute()
        or not isinstance(retrieval_input_dir, str)
        or not Path(retrieval_input_dir).is_absolute()
        or not isinstance(retrieval_input_receipt_sha256, str)
        or not _SHA256_RE.fullmatch(retrieval_input_receipt_sha256)
    ):
        raise ValueError("retrieval preflight destinations must be absolute")
    return receipt


def _publish_preflight(output_dir: Path, receipt: Mapping[str, object]) -> None:
    output = Path(output_dir)
    if not output.name or output.name in {".", ".."}:
        raise ValueError("preflight output path is unsafe")
    try:
        parent_fd = _open_directory_no_symlinks(output.parent)
    except OSError as exc:
        raise ValueError("preflight output parent is missing or unsafe") from exc
    staging_name = f".{output.name}.staging-{uuid.uuid4().hex}"
    staging_fd: int | None = None
    published = False
    try:
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"create-only preflight already exists: {output}")
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        source = _pretty_bytes(receipt)
        file_fd = os.open(
            "receipt.json",
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=staging_fd,
        )
        try:
            view = memoryview(source)
            while view:
                written = os.write(file_fd, view)
                view = view[written:]
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.fsync(staging_fd)
        _rename_noreplace_at(parent_fd, staging_name, output.name)
        published = True
        os.fsync(parent_fd)
    finally:
        if staging_fd is not None:
            if not published:
                try:
                    os.unlink("receipt.json", dir_fd=staging_fd)
                except FileNotFoundError:
                    pass
            os.close(staging_fd)
        if not published:
            try:
                os.rmdir(staging_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _rename_noreplace_at(
    parent_fd: int, staging_name: str, destination_name: str
) -> None:
    if any(
        not name or name in {".", ".."} or "/" in name
        for name in (staging_name, destination_name)
    ):
        raise ValueError("retrieval publish name differs")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic create-only rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(staging_name),
        parent_fd,
        os.fsencode(destination_name),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            f"create-only retrieval publication exists: {destination_name}"
        )
    raise OSError(error, os.strerror(error), destination_name)


def _audit_retrieval_cache_rows(
    accepted_o1_rows: Sequence[Mapping[str, object]],
    *,
    endpoint: object = DEFAULT_ENDPOINT,
    index_id: object = INDEX_ID,
    cache_root: Path = SHARED_CACHE_DIR,
    retrieval_output_dir: Path,
    cache_loader: Callable[[dict[str, object]], object] | None = None,
    retrieval_input_dir: Path = Path("/var/tmp/adaptive-v2-fixture-inputs"),
    retrieval_input_receipt_sha256: str = "f" * 64,
) -> dict[str, object]:
    """Audit exact cache identities without constructing transport or sessions."""

    jobs = _build_retrieval_jobs_from_inputs(
        accepted_o1_rows, endpoint=endpoint, index_id=index_id
    )
    input_root = Path(retrieval_input_dir).absolute()
    if not _SHA256_RE.fullmatch(retrieval_input_receipt_sha256):
        raise ValueError("retrieval input receipt hash differs")
    root = Path(cache_root).absolute()
    final_output = Path(retrieval_output_dir).absolute()
    statuses: list[dict[str, object]] = []
    cache_store: _SecureCacheStore | None = None
    if cache_loader is None:
        cache_store = _SecureCacheStore(root)
        cache_loader = cache_store.load
    try:
        for job in jobs:
            statuses.append(_cache_status(cache_loader(job), job))
    finally:
        if cache_store is not None:
            cache_store.close()
    hits = sum(status["hit"] is True for status in statuses)
    misses = len(statuses) - hits
    cached_bytes = sum(int(status["raw_bytes"]) for status in statuses if status["hit"] is True)
    receipt: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "retrieval_input_dir": str(input_root),
        "retrieval_input_receipt_sha256": retrieval_input_receipt_sha256,
        "planned_request_count": len(jobs),
        "verified_cache_hits": hits,
        "verified_cache_misses": misses,
        "expected_raw_rows": len(jobs) * RETRIEVAL_HITS,
        "observed_cached_raw_bytes": cached_bytes,
        "estimated_new_raw_bytes": misses * ESTIMATED_RAW_BYTES_PER_REQUEST,
        "estimated_total_raw_bytes": cached_bytes + misses * ESTIMATED_RAW_BYTES_PER_REQUEST,
        "primary_external_attempts": misses,
        "maximum_external_attempts": misses,
        "hits": RETRIEVAL_HITS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "transport_retry_count": TRANSPORT_RETRY_COUNT,
        "rate_limiter": dict(RATE_LIMITER_IDENTITY),
        "endpoint": _validated_endpoint(endpoint),
        "index_id": str(index_id),
        "cache_root": str(root),
        "output_dir": str(final_output),
        "cache_audit": statuses,
        "request_inventory_sha256": canonical_sha256(jobs),
        "requests": jobs,
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "paid_call_count": 0,
    }
    _verify_receipt(receipt)
    return receipt


def audit_retrieval_cache(
    retrieval_input_dir: Path,
    *,
    endpoint: object = DEFAULT_ENDPOINT,
    index_id: object = INDEX_ID,
    cache_root: Path = SHARED_CACHE_DIR,
    retrieval_output_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Audit only the authenticated frozen Task 4→5 retrieval input bundle."""

    authenticated = load_authenticated_retrieval_inputs(Path(retrieval_input_dir))
    inputs = authenticated.get("inputs")
    receipt_sha256 = authenticated.get("receipt_sha256")
    if (
        not isinstance(inputs, list)
        or not isinstance(receipt_sha256, str)
        or not _SHA256_RE.fullmatch(receipt_sha256)
    ):
        raise ValueError("retrieval input inventory differs")
    receipt = _audit_retrieval_cache_rows(
        inputs,
        endpoint=endpoint,
        index_id=index_id,
        cache_root=cache_root,
        retrieval_output_dir=retrieval_output_dir,
        retrieval_input_dir=Path(retrieval_input_dir),
        retrieval_input_receipt_sha256=receipt_sha256,
    )
    if output_dir is not None:
        _publish_preflight(Path(output_dir), receipt)
    return receipt


def _capture_preflight_source(path: Path) -> bytes:
    expected = {"receipt.json"}
    try:
        descriptor = _open_directory_no_symlinks(path)
        try:
            before = os.fstat(descriptor)
            names_before = set(os.listdir(descriptor))
            if names_before != expected:
                raise OSError("preflight inventory differs")
            source = _read_stable_regular_at(
                descriptor, "receipt.json", require_single_link=True
            )
            names_after = set(os.listdir(descriptor))
            after = os.fstat(descriptor)
            if names_after != names_before or (
                before.st_dev,
                before.st_ino,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise OSError("preflight changed during capture")
            return source
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ValueError("retrieval preflight is missing or unsafe") from exc


def _parse_preflight_source(source: bytes) -> dict[str, object]:
    try:
        receipt = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("retrieval preflight receipt is invalid") from exc
    if not isinstance(receipt, dict) or source != _pretty_bytes(receipt):
        raise ValueError("retrieval preflight receipt is not canonical")
    return _verify_receipt(receipt)


def _verify_retrieval_preflight_sources(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    input_dir = receipt.get("retrieval_input_dir")
    expected_receipt_sha256 = receipt.get("retrieval_input_receipt_sha256")
    if (
        not isinstance(input_dir, str)
        or not Path(input_dir).is_absolute()
        or not isinstance(expected_receipt_sha256, str)
        or not _SHA256_RE.fullmatch(expected_receipt_sha256)
    ):
        raise ValueError("retrieval input binding differs")
    authenticated = load_authenticated_retrieval_inputs(Path(input_dir))
    rows = authenticated.get("inputs")
    observed_receipt_sha256 = authenticated.get("receipt_sha256")
    if (
        not isinstance(rows, list)
        or observed_receipt_sha256 != expected_receipt_sha256
    ):
        raise ValueError("retrieval input receipt binding differs")
    expected_jobs = _build_retrieval_jobs_from_inputs(
        rows,
        endpoint=receipt.get("endpoint"),
        index_id=receipt.get("index_id"),
    )
    observed_jobs = receipt.get("requests")
    if (
        not isinstance(observed_jobs, list)
        or _canonical_bytes(observed_jobs) != _canonical_bytes(expected_jobs)
        or receipt.get("request_inventory_sha256")
        != canonical_sha256(expected_jobs)
    ):
        raise ValueError("retrieval jobs differ from authenticated input replay")
    return dict(receipt)


def verify_retrieval_preflight(preflight_dir: Path) -> dict[str, object]:
    receipt = _parse_preflight_source(
        _capture_preflight_source(Path(preflight_dir))
    )
    return _verify_retrieval_preflight_sources(receipt)


def _capture_retrieval_approval(path: Path) -> tuple[dict[str, object], str]:
    try:
        source = _capture_regular_file_no_symlinks(Path(path))
        value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError("retrieval approval required") from exc
    required = {
        "schema_version",
        "stage",
        "preflight_sha256",
        "planned_request_count",
        "primary_external_attempts",
        "maximum_external_attempts",
        "transport_retry_count",
        "cache_root",
        "output_dir",
        "approved",
    }
    if (
        not isinstance(value, dict)
        or source != _pretty_bytes(value)
        or not required <= set(value)
        or value.get("schema_version")
        != "adaptive-obligation-v2-retrieval-approval-v1"
        or value.get("stage") != "retrieval"
        or value.get("approved") is not True
        or not isinstance(value.get("preflight_sha256"), str)
        or not _SHA256_RE.fullmatch(str(value["preflight_sha256"]))
        or any(
            type(value.get(name)) is not int
            for name in (
                "planned_request_count",
                "primary_external_attempts",
                "maximum_external_attempts",
                "transport_retry_count",
            )
        )
        or value.get("transport_retry_count") != 0
        or not isinstance(value.get("cache_root"), str)
        or not Path(str(value["cache_root"])).is_absolute()
        or not isinstance(value.get("output_dir"), str)
        or not Path(str(value["output_dir"])).is_absolute()
    ):
        raise PermissionError("retrieval approval required")
    return value, _sha256(source)


def _capture_retrieval_preflight(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, object], bytes]:
    source = _capture_preflight_source(path)
    if _sha256(source) != expected_sha256:
        raise PermissionError("retrieval approval required")
    return _parse_preflight_source(source), source


def _verify_approval_against_preflight(
    approval: Mapping[str, object], receipt: Mapping[str, object]
) -> None:
    for name in (
        "planned_request_count",
        "primary_external_attempts",
        "maximum_external_attempts",
        "transport_retry_count",
        "cache_root",
        "output_dir",
    ):
        if approval.get(name) != receipt.get(name):
            raise PermissionError("retrieval approval required")


def _current_cache(
    receipt: Mapping[str, object], cache: _SecureCacheStore
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    hits: list[dict[str, object]] = []
    misses: list[dict[str, object]] = []
    observed: list[dict[str, object]] = []
    jobs = receipt["requests"]
    assert isinstance(jobs, list)
    for raw_job in jobs:
        job = dict(raw_job)
        cached = cache.load(job)
        observed.append(_cache_status(cached, job))
        (hits if cached is not None else misses).append(
            {"job": job, "cache": cached} if cached is not None else {"job": job}
        )
    if observed != receipt.get("cache_audit"):
        raise ValueError("retrieval cache changed after frozen preflight")
    return hits, misses


def _open_dir_at(parent_fd: int, name: str, *, create: bool) -> int:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("secure directory name differs")
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        os.close(descriptor)
        raise ValueError("secure directory identity differs")
    return descriptor


def _write_exclusive_at(directory_fd: int, name: str, source: bytes) -> None:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("secure output file name differs")
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(source)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short secure write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


class _SecureRunLedger:
    _SUBDIRS = (
        "attempts",
        "raw",
        "metadata",
        "candidates",
        "outcomes",
        "cache_hits",
    )

    def __init__(self, output: Path) -> None:
        self.output = Path(output)
        self.parent_fd = _open_directory_no_symlinks(self.output.parent)
        self.root_fd: int | None = None
        self.subdirs: dict[str, int] = {}
        try:
            os.mkdir(self.output.name, mode=0o700, dir_fd=self.parent_fd)
            os.fsync(self.parent_fd)
        except FileExistsError:
            os.close(self.parent_fd)
            raise FileExistsError(
                f"create-only retrieval output exists: {self.output}"
            ) from None
        try:
            self.root_fd = _open_dir_at(
                self.parent_fd, self.output.name, create=False
            )
            for name in self._SUBDIRS:
                self.subdirs[name] = _open_dir_at(self.root_fd, name, create=True)
            os.fsync(self.root_fd)
        except BaseException:
            self.close()
            raise

    def write_bytes(self, directory: str | None, name: str, source: bytes) -> None:
        descriptor = self.root_fd if directory is None else self.subdirs[directory]
        assert descriptor is not None
        _write_exclusive_at(descriptor, name, source)

    def write_json(self, directory: str | None, name: str, value: object) -> None:
        self.write_bytes(directory, name, _pretty_bytes(value))

    def close(self) -> None:
        for descriptor in self.subdirs.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.subdirs.clear()
        if self.root_fd is not None:
            try:
                os.fsync(self.root_fd)
                os.close(self.root_fd)
            except OSError:
                pass
            self.root_fd = None
        if hasattr(self, "parent_fd"):
            try:
                os.fsync(self.parent_fd)
                os.close(self.parent_fd)
            except OSError:
                pass


def _claim_output_root(output: Path) -> _SecureRunLedger:
    return _SecureRunLedger(output)


def _candidate_with_provenance(
    job: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    raw_response_sha256: str,
) -> dict[str, object]:
    """Bind every persisted candidate to its local scoring and source lineage."""

    frozen_input = job.get("retrieval_input")
    if not isinstance(frozen_input, Mapping):
        raise ValueError("retrieval candidate source lineage is missing")
    return {
        "schema_version": "adaptive-obligation-v2-candidate-v1",
        "request_key": job["request_key"],
        "request_identity": job["request_identity"],
        "query_text": job["query_text"],
        "query_sha256": job["query_sha256"],
        "raw_response_sha256": raw_response_sha256,
        "topic_id": job["topic_id"],
        "parent_id": job["parent_id"],
        "accepted_o1_id": job["accepted_o1_id"],
        "accepted_o1_sha256": job["accepted_o1_sha256"],
        "retrieval_input": dict(frozen_input),
        "docid": candidate["docid"],
        "rank": candidate["rank"],
        "score": candidate["score"],
        "text": candidate["text"],
        "text_sha256": candidate["text_sha256"],
    }


class RateLimitedO1Transport:
    """Tracked requests-ratelimiter transport with no hidden retry path."""

    one_shot_no_retry = True
    request_start_interval_seconds = REQUEST_START_INTERVAL_SECONDS
    timeout_seconds = TIMEOUT_SECONDS

    def __init__(
        self,
        allowed_jobs: Sequence[Mapping[str, object]],
        *,
        api_token: str | None,
    ) -> None:
        jobs = [dict(_verify_job(job)) for job in allowed_jobs]
        self._allowed = {str(job["request_key"]): job for job in jobs}
        if len(self._allowed) != len(jobs):
            raise ValueError("transport allowlist contains duplicate requests")
        endpoints = {str(job["endpoint"]) for job in jobs}
        if len(endpoints) != 1:
            raise ValueError("transport allowlist must use one endpoint")
        self.endpoint = next(iter(endpoints))
        self.api_token = api_token
        config = RemotePyseriniConfig(
            index_url=self.endpoint,
            api_token=api_token,
            hits=RETRIEVAL_HITS,
            queries=(),
            min_interval_seconds=REQUEST_START_INTERVAL_SECONDS,
            burst=1,
            limiter_state_path=LIMITER_STATE_PATH,
        )
        self.session = rate_limited_session(config)

    def __call__(self, job: Mapping[str, object]) -> RawTransportResponse:
        allowed = self._allowed.get(str(job.get("request_key")))
        if allowed != dict(job):
            raise ValueError("request is outside the frozen transport allowlist")
        started = time.monotonic()
        response = self.session.get(
            self.endpoint,
            params={"query": job["query_text"], "hits": str(RETRIEVAL_HITS)},
            headers={
                "Accept": "application/json",
                **(
                    {"Authorization": f"Bearer {self.api_token}"}
                    if self.api_token is not None
                    else {}
                ),
            },
            timeout=TIMEOUT_SECONDS,
            allow_redirects=False,
        )
        elapsed = getattr(response, "elapsed", None)
        elapsed_seconds = (
            float(elapsed.total_seconds())
            if elapsed is not None and hasattr(elapsed, "total_seconds")
            else time.monotonic() - started
        )
        return RawTransportResponse(
            status=int(response.status_code),
            headers=dict(response.headers),
            body=bytes(response.content),
            elapsed_seconds=elapsed_seconds,
        )


def _build_approved_transport(
    jobs: Sequence[Mapping[str, object]], *, api_token: str | None
) -> RetrievalTransport:
    return RateLimitedO1Transport(jobs, api_token=api_token)


def execute_retrieval(
    *,
    preflight_dir: Path,
    approval_path: Path,
    api_token: str | None = None,
) -> dict[str, object]:
    """Execute an approved immutable request inventory exactly once."""

    approval, approval_sha256 = _capture_retrieval_approval(Path(approval_path))
    receipt, receipt_source = _capture_retrieval_preflight(
        Path(preflight_dir), expected_sha256=str(approval["preflight_sha256"])
    )
    _verify_approval_against_preflight(approval, receipt)
    _verify_retrieval_preflight_sources(receipt)
    return _execute_verified_retrieval(
        receipt=receipt,
        receipt_source=receipt_source,
        approval_sha256=approval_sha256,
        api_token=api_token,
    )


def _execute_verified_retrieval(
    *,
    receipt: Mapping[str, object],
    receipt_source: bytes,
    approval_sha256: str,
    api_token: str | None = None,
) -> dict[str, object]:
    """Execute only after the public boundary authenticates every source."""

    root = Path(str(receipt["cache_root"]))
    output = Path(str(receipt["output_dir"]))
    cache_store = _SecureCacheStore(root)
    writer: _SecureRunLedger | None = None
    try:
        cache_hits, cache_misses = _current_cache(receipt, cache_store)
        miss_jobs = [row["job"] for row in cache_misses]
        transport: RetrievalTransport | None = None
        if miss_jobs:
            transport = _build_approved_transport(miss_jobs, api_token=api_token)
            if getattr(transport, "one_shot_no_retry", None) is not True:
                raise ValueError("retrieval transport must be one-shot no-retry")
            if (
                getattr(transport, "request_start_interval_seconds", None)
                != REQUEST_START_INTERVAL_SECONDS
                or getattr(transport, "timeout_seconds", None) != TIMEOUT_SECONDS
            ):
                raise ValueError("retrieval transport limiter or timeout differs")
        writer = _claim_output_root(output)
        combined: list[dict[str, object]] = []
        external_attempts = 0
        cached_by_key = {
            str(row["job"]["request_key"]): row for row in cache_hits
        }
        jobs = receipt["requests"]
        assert isinstance(jobs, list)
        for order, raw_job in enumerate(jobs):
            job = dict(raw_job)
            key = str(job["request_key"])
            cached_row = cached_by_key.get(key)
            if cached_row is not None:
                cached = cached_row["cache"]
                assert isinstance(cached, Mapping)
                raw = cached["raw"]
                candidates = cached["candidates"]
                assert isinstance(raw, bytes) and isinstance(candidates, tuple)
                writer.write_json(
                    "cache_hits",
                    f"{key}.json",
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "request_key": key,
                        "request_identity": job["request_identity"],
                        "raw_sha256": cached["raw_sha256"],
                        "candidate_count": RETRIEVAL_HITS,
                        "external_attempts": 0,
                    },
                )
                writer.write_bytes("raw", f"{key}.body", raw)
                raw_sha256 = str(cached["raw_sha256"])
                writer.write_json(
                    "outcomes",
                    f"{key}.json",
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "request_key": key,
                        "status": "cache_hit",
                        "candidate_count": RETRIEVAL_HITS,
                        "raw_sha256": raw_sha256,
                    },
                )
            else:
                assert transport is not None
                writer.write_json(
                    "attempts",
                    f"{key}.json",
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "request_key": key,
                        "request_identity": job["request_identity"],
                        "query_text": job["query_text"],
                        "attempt_ordinal": 1,
                        "transport_retry_count": 0,
                        "manifest_order": order,
                    },
                )
                external_attempts += 1
                try:
                    response = transport(job)
                except Exception as exc:
                    writer.write_json(
                        "outcomes",
                        f"{key}.json",
                        {
                            "schema_version": LEDGER_SCHEMA_VERSION,
                            "request_key": key,
                            "status": "failure",
                            "failure_type": "transport_exception",
                            "message": f"{type(exc).__name__}: {exc}",
                            "raw_sha256": None,
                        },
                    )
                    raise
                raw = response.body
                writer.write_bytes("raw", f"{key}.body", raw)
                raw_sha256 = _sha256(raw)
                writer.write_json(
                    "metadata",
                    f"{key}.json",
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "request_key": key,
                        "http_status": response.status,
                        "headers": dict(response.headers),
                        "elapsed_seconds": float(response.elapsed_seconds),
                        "raw_sha256": raw_sha256,
                    },
                )
                if response.status != 200:
                    message = f"HTTP status {response.status} is not successful"
                    writer.write_json(
                        "outcomes",
                        f"{key}.json",
                        {
                            "schema_version": LEDGER_SCHEMA_VERSION,
                            "request_key": key,
                            "status": "failure",
                            "failure_type": "http_error",
                            "message": message,
                            "raw_sha256": raw_sha256,
                        },
                    )
                    raise ValueError(message)
                try:
                    candidates = _normalize_response(raw)
                except ValueError as exc:
                    writer.write_json(
                        "outcomes",
                        f"{key}.json",
                        {
                            "schema_version": LEDGER_SCHEMA_VERSION,
                            "request_key": key,
                            "status": "failure",
                            "failure_type": "response_validation_error",
                            "message": str(exc),
                            "raw_sha256": raw_sha256,
                        },
                    )
                    raise
                cache_store.store(job, raw, candidates)
                writer.write_json(
                    "outcomes",
                    f"{key}.json",
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "request_key": key,
                        "status": "success",
                        "candidate_count": RETRIEVAL_HITS,
                        "raw_sha256": raw_sha256,
                    },
                )
            provenance_candidates = [
                _candidate_with_provenance(
                    job, candidate, raw_response_sha256=raw_sha256
                )
                for candidate in candidates
            ]
            writer.write_json(
                "candidates", f"{key}.json", provenance_candidates
            )
            combined.extend(provenance_candidates)

        writer.write_bytes(
            None,
            "candidates.jsonl",
            b"".join(_canonical_bytes(row) + b"\n" for row in combined),
        )
        summary: dict[str, object] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "preflight_sha256": _sha256(receipt_source),
            "approval_sha256": approval_sha256,
            "cache_root": str(root),
            "output_dir": str(output),
            "planned_request_count": len(jobs),
            "cache_hits": len(cache_hits),
            "external_attempts": external_attempts,
            "candidate_rows": len(combined),
            "transport_retry_count": 0,
            "qrels_opened": False,
        }
        writer.write_json(None, "summary.json", summary)
        return summary
    finally:
        if writer is not None:
            writer.close()
        cache_store.close()


__all__ = [
    "INDEX_ID",
    "MAX_ACCEPTED_O1_PER_TOPIC",
    "MAX_RETRIEVAL_REQUESTS",
    "RATE_LIMITER_IDENTITY",
    "REQUEST_START_INTERVAL_SECONDS",
    "RETRIEVAL_HITS",
    "RETRIEVER_VERSION",
    "RateLimitedO1Transport",
    "TIMEOUT_SECONDS",
    "TRANSPORT_RETRY_COUNT",
    "audit_retrieval_cache",
    "build_retrieval_jobs",
    "execute_retrieval",
    "render_o1_bm25_query",
    "verify_retrieval_preflight",
]
