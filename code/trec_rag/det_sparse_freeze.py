"""Immutable artifact sealing, qrels firewall, and frozen pilot decisions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from trec_rag.det_sparse_config import (
    ANALYZER_FINGERPRINT_SHA256,
    ARM_NAMES,
    EVALUATION_METRICS,
    EXPERIMENT_ID,
    PILOT_TOPIC_IDS,
)
from trec_rag.evaluation import evaluate_ranked, parse_qrels_bytes
from trec_rag.pipeline_models import RankedCandidate
from trec_rag.query_analyzer import QueryAnalyzer, RemoteLuceneQueryAnalyzer


FREEZE_SCHEMA_VERSION = "det_sparse_freeze_manifest_v1"
FINAL_ARTIFACT_KEYS = (
    "config",
    "preflight_freeze",
    "query_registry",
    "ledger_manifest",
    "ledger_validation",
    "global_budget_receipts",
    "prf_artifacts",
    "arm_plans",
    "runtime",
    "run_summary",
    "ranking_O",
    "ranking_F",
    "ranking_E",
    "ranking_FE",
)
_EXPECTED_REQUIRED_ARTIFACTS = {
    "config": "execution/config.yaml",
    "preflight_freeze": "preflight/pre_retrieval_freeze.json",
    "query_registry": "execution/query_registry.json",
    "ledger_manifest": "execution/ledger/ledger.json",
    "ledger_validation": "execution/ledger_validation.json",
    "global_budget_receipts": "execution/global_budget_receipts.json",
    "prf_artifacts": "execution/prf_artifacts.json",
    "arm_plans": "execution/arm_plans.json",
    "runtime": "execution/runtime.json",
    "run_summary": "execution/run_summary.json",
    "ranking_O": "execution/rankings/O.jsonl",
    "ranking_F": "execution/rankings/F.jsonl",
    "ranking_E": "execution/rankings/E.jsonl",
    "ranking_FE": "execution/rankings/FE.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def rankings_sha256(
    rankings: Mapping[str, list[RankedCandidate]],
) -> dict[str, str]:
    if tuple(rankings) != ARM_NAMES:
        raise ValueError(f"rankings must be ordered as the frozen arms {ARM_NAMES!r}")
    result: dict[str, str] = {}
    for arm in ARM_NAMES:
        ordered = sorted(
            rankings[arm],
            key=lambda row: (row.topic_id, row.rank, row.docid),
        )
        observed_topics = {row.topic_id for row in ordered}
        if observed_topics != set(PILOT_TOPIC_IDS):
            raise ValueError(f"ranking {arm} does not cover the exact four pilot topics")
        for topic_id in PILOT_TOPIC_IDS:
            topic_rows = [row for row in ordered if row.topic_id == topic_id]
            if len(topic_rows) != 100:
                raise ValueError(f"ranking {arm}/{topic_id} depth is not exactly 100")
            if [row.rank for row in topic_rows] != list(range(1, len(topic_rows) + 1)):
                raise ValueError(f"ranking {arm}/{topic_id} ranks are not contiguous")
            if len({row.docid for row in topic_rows}) != len(topic_rows):
                raise ValueError(f"ranking {arm}/{topic_id} has duplicate docids")
            if any(
                not row.docid
                or isinstance(row.score, bool)
                or not math.isfinite(float(row.score))
                for row in topic_rows
            ):
                raise ValueError(f"ranking {arm}/{topic_id} contains invalid values")
        payload = [asdict(row) for row in ordered]
        result[arm] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return result


def _create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def create_freeze_manifest(
    output_path: Path,
    *,
    artifact_root: Path,
    artifacts: Sequence[Path],
    qrels_path: Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Seal all pre-evaluation artifacts without reading the qrels file."""

    root = artifact_root.resolve()
    output = output_path.resolve()
    qrels = qrels_path.resolve()
    if not artifacts:
        raise ValueError("freeze manifest requires at least one artifact")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_path in artifacts:
        path = raw_path.resolve()
        if path == output:
            raise ValueError("freeze manifest cannot include itself")
        if path == qrels:
            raise ValueError("qrels must not be opened or hashed before the freeze")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"freeze artifact is outside artifact_root: {path}") from exc
        if relative in seen:
            raise ValueError(f"duplicate freeze artifact: {relative}")
        seen.add(relative)
        if not path.is_file():
            raise ValueError(f"freeze artifact is missing or not a file: {path}")
        rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest: dict[str, object] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "artifact_root": str(root),
        "artifacts": sorted(rows, key=lambda row: str(row["path"])),
        "qrels_path": str(qrels),
        "qrels_opened_before_freeze": False,
        "metadata": dict(metadata),
    }
    _create_only(output, _canonical_json(manifest))
    return manifest


def create_evaluation_freeze(
    output_path: Path,
    *,
    artifact_root: Path,
    required_artifacts: Mapping[str, Path],
    additional_artifacts: Sequence[Path],
    qrels_path: Path,
    rankings: Mapping[str, list[RankedCandidate]],
    run_summary: Mapping[str, object],
) -> dict[str, object]:
    """Seal the complete execution boundary required before qrels access."""

    if tuple(required_artifacts) != FINAL_ARTIFACT_KEYS:
        raise ValueError(
            f"required artifacts must be ordered as {FINAL_ARTIFACT_KEYS!r}"
        )
    root = artifact_root.resolve()
    required_relative: dict[str, str] = {}
    for name, raw_path in required_artifacts.items():
        path = raw_path.resolve()
        try:
            required_relative[name] = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"required artifact {name} is outside artifact_root") from exc
    metadata = {
        **dict(run_summary),
        "required_artifacts": required_relative,
        "ranking_sha256": rankings_sha256(rankings),
    }
    _validate_final_metadata(metadata)
    return create_freeze_manifest(
        output_path,
        artifact_root=root,
        artifacts=(*required_artifacts.values(), *additional_artifacts),
        qrels_path=qrels_path,
        metadata=metadata,
    )


def verify_freeze_manifest(
    manifest_path: Path,
    *,
    artifact_root: Path,
    expected_qrels_path: Path,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("freeze manifest must be an object")
    if manifest.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise ValueError("freeze manifest schema version mismatch")
    root = artifact_root.resolve()
    if manifest.get("artifact_root") != str(root):
        raise ValueError("freeze manifest artifact root mismatch")
    if manifest.get("qrels_path") != str(expected_qrels_path.resolve()):
        raise ValueError("freeze manifest qrels identity mismatch")
    if manifest.get("qrels_opened_before_freeze") is not False:
        raise ValueError("freeze manifest does not preserve the qrels firewall")
    raw_rows = manifest.get("artifacts")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("freeze manifest lacks artifacts")
    seen: set[str] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("freeze manifest artifact row must be an object")
        relative = raw_row.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise ValueError("freeze manifest artifact path is invalid or duplicated")
        seen.add(relative)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("freeze artifact path escapes artifact_root") from exc
        if not path.is_file():
            raise ValueError(f"frozen artifact is missing: {relative}")
        if raw_row.get("size") != path.stat().st_size:
            raise ValueError(f"frozen artifact size mismatch: {relative}")
        if raw_row.get("sha256") != sha256_file(path):
            raise ValueError(f"frozen artifact hash mismatch: {relative}")
    return manifest


def _valid_git_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_final_metadata(metadata: Mapping[str, object]) -> None:
    if metadata.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("final freeze experiment identity mismatch")
    if metadata.get("topic_ids") != list(PILOT_TOPIC_IDS):
        raise ValueError("final freeze topic identity/order mismatch")
    if metadata.get("arms") != list(ARM_NAMES):
        raise ValueError("final freeze arm identity/order mismatch")
    if metadata.get("mechanical_valid") is not True:
        raise ValueError("final freeze mechanical gate did not pass")
    if metadata.get("retrieval_complete") is not True:
        raise ValueError("final freeze retrieval is incomplete")
    if metadata.get("fallback_topic_ids") != []:
        raise ValueError("final freeze contains fallback topics")
    if metadata.get("qrels_opened") is not False:
        raise ValueError("final freeze violated the qrels firewall")
    if metadata.get("model_calls") != 0 or metadata.get("reranker_calls") != 0:
        raise ValueError("final freeze contains model or reranker calls")
    if metadata.get("hits") != 100 or metadata.get("index_id") != "climbmix-400b":
        raise ValueError("final freeze retrieval identity/depth mismatch")
    if metadata.get("analyzer_fingerprint_sha256") != ANALYZER_FINGERPRINT_SHA256:
        raise ValueError("final freeze analyzer fingerprint mismatch")
    if metadata.get("endpoint_path_index_verified") is not True:
        raise ValueError("final freeze endpoint path/index identity was not verified")
    if metadata.get("index_revision") != "hosted_climbmix_unknown_revision":
        raise ValueError("final freeze must retain the unknown hosted index revision caveat")
    if metadata.get("cache_scope") != "fresh_run_local":
        raise ValueError("formal pilot must use a fresh run-local cache scope")

    per_topic = metadata.get("per_topic_external_calls")
    if not isinstance(per_topic, Mapping) or list(per_topic) != list(PILOT_TOPIC_IDS):
        raise ValueError("final freeze lacks ordered per-topic call counts")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9
        for value in per_topic.values()
    ):
        raise ValueError("final freeze exceeds the nine-call per-topic ceiling")
    external_calls = metadata.get("external_calls")
    if (
        isinstance(external_calls, bool)
        or not isinstance(external_calls, int)
        or external_calls != sum(int(value) for value in per_topic.values())
        or external_calls > 36
    ):
        raise ValueError("final freeze external call count is invalid")
    planned_requests = metadata.get("planned_requests")
    if (
        isinstance(planned_requests, bool)
        or not isinstance(planned_requests, int)
        or planned_requests != external_calls
    ):
        raise ValueError("fresh formal run planned/external counts differ")
    if metadata.get("cache_hits") != 0:
        raise ValueError("formal pilot may not reuse pre-existing cache entries")
    budget_root = metadata.get("global_budget_root")
    if not isinstance(budget_root, str) or not Path(budget_root).is_absolute():
        raise ValueError("final freeze lacks an absolute shared global budget root")
    if metadata.get("global_budget_receipts") != external_calls:
        raise ValueError("global budget receipts differ from external call count")

    ledger = metadata.get("ledger_validation")
    if not isinstance(ledger, Mapping):
        raise ValueError("final freeze lacks ledger validation")
    if (
        ledger.get("reservations") != external_calls
        or ledger.get("successes") != external_calls
        or ledger.get("failures") != 0
        or ledger.get("pending") != 0
        or ledger.get("cache_hits") != 0
        or ledger.get("external_calls") != external_calls
        or ledger.get("planned_requests") != external_calls
        or ledger.get("per_topic_external_calls") != dict(per_topic)
    ):
        raise ValueError("final freeze ledger is incomplete or inconsistent")

    source = metadata.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("source_tree_clean") is not True
        or not _valid_git_hash(source.get("commit"))
        or not _valid_git_hash(source.get("tree"))
    ):
        raise ValueError("final freeze source provenance is invalid")
    runtime = metadata.get("runtime")
    if not isinstance(runtime, Mapping) or not runtime.get("python"):
        raise ValueError("final freeze runtime provenance is missing")
    boundary = metadata.get("runtime_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("transport_version")
        != "det_sparse_https_one_shot_no_redirect_v1"
        or boundary.get("transport_class")
        != "trec_rag.det_sparse_transport.FrozenHttpsRetrievalTransport"
        or boundary.get("endpoint_url")
        != "https://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
        or boundary.get("one_shot_no_retry") is not True
        or boundary.get("redirects_allowed") is not False
        or boundary.get("injected_opener") is not False
        or boundary.get("analyzer_client_class")
        != "trec_rag.query_analyzer.RemoteLuceneQueryAnalyzer"
        or boundary.get("analyzer_url") != "http://127.0.0.1:18081"
        or not isinstance(boundary.get("analyzer_conformance_probe_sha256"), Mapping)
        or not boundary["analyzer_conformance_probe_sha256"]
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in boundary["analyzer_conformance_probe_sha256"].values()
        )
    ):
        raise ValueError("final freeze runtime/network/analyzer boundary is invalid")
    required = metadata.get("required_artifacts")
    if not isinstance(required, Mapping) or set(required) != set(FINAL_ARTIFACT_KEYS):
        raise ValueError("final freeze required-artifact map is incomplete")
    if any(not isinstance(path, str) or not path for path in required.values()):
        raise ValueError("final freeze required-artifact path is invalid")
    hashes = metadata.get("ranking_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(ARM_NAMES):
        raise ValueError("final freeze ranking hashes are incomplete")
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in hashes.values()
    ):
        raise ValueError("final freeze ranking hash is invalid")


def _read_json_object(path: Path, owner: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{owner} is not readable canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a JSON object")
    return value


def _read_json_array(path: Path, owner: str) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{owner} is not readable canonical JSON") from exc
    if not isinstance(value, list):
        raise ValueError(f"{owner} must be a JSON array")
    return value


def _json_normalize(value: object) -> object:
    return json.loads(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _required_paths(
    metadata: Mapping[str, object],
    root: Path,
) -> dict[str, Path]:
    raw = metadata.get("required_artifacts")
    if not isinstance(raw, Mapping) or dict(raw) != _EXPECTED_REQUIRED_ARTIFACTS:
        raise ValueError("final freeze required paths differ from the fixed layout")
    result: dict[str, Path] = {}
    for name in FINAL_ARTIFACT_KEYS:
        relative = _EXPECTED_REQUIRED_ARTIFACTS[name]
        unresolved = root / relative
        path = unresolved.resolve()
        if unresolved.is_symlink() or path != unresolved.absolute():
            raise ValueError(f"required artifact {name} uses a symlink or path alias")
        if not path.is_file():
            raise ValueError(f"required artifact {name} is missing")
        result[name] = path
    if len(set(result.values())) != len(result):
        raise ValueError("required artifact paths are not distinct")
    return result


def _canonical_registry_rows(
    variants: Sequence[object],
    *,
    stage: str,
) -> list[dict[str, object]]:
    from trec_rag.pipeline_models import QueryVariant

    typed: list[QueryVariant] = []
    for value in variants:
        if not isinstance(value, QueryVariant):
            raise TypeError("canonical registry input must contain QueryVariant rows")
        typed.append(value)
    by_topic_text: dict[tuple[str, str], list[QueryVariant]] = {}
    order: list[tuple[str, str]] = []
    for variant in typed:
        key = (variant.topic_id, variant.query_text)
        if key not in by_topic_text:
            by_topic_text[key] = []
            order.append(key)
        by_topic_text[key].append(variant)
    ordinals: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for topic_id, query_text in order:
        ordinals[topic_id] = ordinals.get(topic_id, 0) + 1
        aliases = by_topic_text[(topic_id, query_text)]
        rows.append(
            {
                "topic_id": topic_id,
                "stage": stage,
                "canonical_query_id": (
                    f"{topic_id}:{stage}:{ordinals[topic_id]:02d}"
                ),
                "canonical_variant_name": aliases[0].variant_name,
                "query_text": query_text,
                "query_sha256": hashlib.sha256(
                    query_text.encode("utf-8")
                ).hexdigest(),
                "logical_variant_aliases": [
                    alias.variant_name for alias in aliases
                ],
            }
        )
    return rows


def _load_frozen_ranking_files(
    required: Mapping[str, Path],
) -> dict[str, list[RankedCandidate]]:
    rankings: dict[str, list[RankedCandidate]] = {}
    exact_fields = {"topic_id", "docid", "rank", "score", "text", "provenance"}
    expected_order = [
        (topic_id, rank)
        for topic_id in PILOT_TOPIC_IDS
        for rank in range(1, 101)
    ]
    for arm in ARM_NAMES:
        path = required[f"ranking_{arm}"]
        rows: list[RankedCandidate] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line:
                raise ValueError(f"ranking {arm} contains an empty JSONL row")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"ranking {arm} row {line_number} is invalid JSON"
                ) from exc
            if not isinstance(raw, dict) or set(raw) != exact_fields:
                raise ValueError(f"ranking {arm} row fields are invalid")
            topic_id = raw["topic_id"]
            docid = raw["docid"]
            rank = raw["rank"]
            score = raw["score"]
            text = raw["text"]
            provenance = raw["provenance"]
            if (
                not isinstance(topic_id, str)
                or not isinstance(docid, str)
                or not docid
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not isinstance(text, str)
                or not isinstance(provenance, list)
                or any(not isinstance(item, dict) for item in provenance)
            ):
                raise ValueError(f"ranking {arm} row values are invalid")
            rows.append(
                RankedCandidate(
                    topic_id=topic_id,
                    docid=docid,
                    rank=rank,
                    score=float(score),
                    text=text,
                    provenance=provenance,
                )
            )
        if [(row.topic_id, row.rank) for row in rows] != expected_order:
            raise ValueError(
                f"ranking {arm} is not exact topic-order depth-100 JSONL"
            )
        rankings[arm] = rows
    rankings_sha256(rankings)
    return rankings


def _validate_endpoint_identity(index_url: object, index_id: str) -> str:
    if not isinstance(index_url, str):
        raise ValueError("run summary lacks the exact retrieval endpoint")
    parsed = urlsplit(index_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("run summary retrieval endpoint is inadmissible")
    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 3 or parts[-3:] != ["v1", index_id, "search"]:
        raise ValueError("run summary endpoint path/index identity mismatch")
    return index_url


def _validate_evaluation_analyzer_boundary(
    query_analyzer: QueryAnalyzer,
    expected_url: str,
) -> None:
    if (
        type(query_analyzer) is not RemoteLuceneQueryAnalyzer
        or query_analyzer.base_url != expected_url
        or query_analyzer.timeout != 10.0
    ):
        raise ValueError(
            "final evaluation requires the exact frozen loopback analyzer client"
        )


def _validate_final_artifact_semantics(
    manifest: Mapping[str, object],
    *,
    artifact_root: Path,
    expected_qrels_path: Path,
    query_analyzer: QueryAnalyzer,
) -> dict[str, object]:
    """Replay the complete qrels-blind pipeline from frozen source evidence."""

    # Local imports avoid det_sparse_preflight -> det_sparse_freeze import cycles.
    from trec_rag.det_sparse_arms import (
        build_arm_plans,
        build_verified_prf_from_ledger,
        fuse_arm,
        retrieval_request_for_query,
    )
    from trec_rag.det_sparse_config import load_det_sparse_config
    from trec_rag.det_sparse_budget import GlobalExternalBudget, global_budget_dir
    from trec_rag.det_sparse_ledger import RetrievalLedger
    from trec_rag.det_sparse_preflight import load_selected_topics, validate_preflight
    from trec_rag.det_sparse_provenance import (
        current_runtime_provenance,
        current_source_provenance,
    )
    from trec_rag.deterministic_sparse import build_deterministic_sparse_plan
    from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate

    root = artifact_root.resolve()
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("final freeze lacks metadata")
    required = _required_paths(metadata, root)
    if Path(str(manifest.get("artifact_root"))).resolve() != root:
        raise ValueError("final semantic validation root mismatch")

    frozen_paths = {
        str(row["path"])
        for row in manifest.get("artifacts", [])
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    for relative in frozen_paths:
        unresolved = root / relative
        if unresolved.is_symlink() or unresolved.resolve() != unresolved.absolute():
            raise ValueError("final freeze contains a symlink or aliased artifact")
        if unresolved.resolve() == expected_qrels_path.resolve():
            raise ValueError("qrels were included in the frozen artifact set")

    preflight_dir = required["preflight_freeze"].parent
    ledger_dir = required["ledger_manifest"].parent
    closure = {
        path.resolve().relative_to(root).as_posix()
        for directory in (preflight_dir, ledger_dir)
        for path in directory.rglob("*")
        if path.is_file() and not path.name.endswith(".lock")
    }
    if not closure.issubset(frozen_paths):
        missing = sorted(closure - frozen_paths)
        raise ValueError("final freeze omits consulted artifacts: " + ", ".join(missing))

    preflight_metadata = _read_json_object(
        preflight_dir / "_preflight.json",
        "preflight metadata",
    )
    original_config_path = preflight_metadata.get("config_path")
    if not isinstance(original_config_path, str):
        raise ValueError("preflight lacks its canonical config path")
    config = load_det_sparse_config(Path(original_config_path))
    if config.output_dir.resolve() != root:
        raise ValueError("canonical config output root differs from final artifact root")
    if config.evaluation.qrels.resolve() != expected_qrels_path.resolve():
        raise ValueError("canonical config qrels differ from the final freeze")
    original_config_hash = sha256_file(config.config_path)
    if (
        sha256_file(required["config"]) != original_config_hash
        or preflight_metadata.get("config_sha256") != original_config_hash
    ):
        raise ValueError("copied, original, and preflight config hashes differ")
    validate_preflight(config, preflight_dir)
    observed_source = current_source_provenance(config.root_dir)
    _validate_evaluation_analyzer_boundary(query_analyzer, config.analyzer.url)

    fingerprint_sha = hashlib.sha256(
        json.dumps(
            query_analyzer.fingerprint.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if fingerprint_sha != config.analyzer.expected_fingerprint_sha256:
        raise ValueError("semantic replay analyzer differs from the frozen analyzer")

    selected_topics = load_selected_topics(config)
    plan_rows = preflight_metadata.get("plans")
    if not isinstance(plan_rows, list) or len(plan_rows) != len(selected_topics):
        raise ValueError("preflight plan rows are incomplete")
    plans = []
    for selected, row in zip(selected_topics, plan_rows):
        if not isinstance(row, dict):
            raise ValueError("preflight plan row is invalid")
        if (
            row.get("topic_id") != selected.topic.id
            or row.get("source_line_number") != selected.source_line_number
            or row.get("source_line_sha256") != selected.source_line_sha256
        ):
            raise ValueError("preflight plan/source record identity mismatch")
        plan = build_deterministic_sparse_plan(
            topic_id=selected.topic.id,
            narrative=selected.topic.narrative,
            query_analyzer=query_analyzer,
        )
        plan_path = preflight_dir / str(row.get("plan_path"))
        stored_plan = _read_json_object(plan_path, "preflight plan")
        if _json_normalize(plan.to_dict()) != stored_plan or row.get(
            "plan_sha256"
        ) != sha256_file(plan_path):
            raise ValueError("preflight plan does not replay from exact source/analyzer")
        if plan.status != "ok":
            raise ValueError("semantic replay produced a fallback plan")
        plans.append(plan)

    summary = _read_json_object(required["run_summary"], "run summary")
    expected_summary = {
        key: value
        for key, value in metadata.items()
        if key not in {"required_artifacts", "ranking_sha256"}
    }
    if summary != expected_summary:
        raise ValueError("run summary differs from final freeze metadata")
    if summary.get("failure") is not None:
        raise ValueError("successful final freeze records an execution failure")
    runtime = _read_json_object(required["runtime"], "runtime provenance")
    if (
        runtime != summary.get("runtime")
        or runtime != current_runtime_provenance(config.root_dir)
    ):
        raise ValueError(
            "runtime artifact differs from the run summary or current environment"
        )
    if (
        summary.get("source") != preflight_metadata.get("source")
        or summary.get("source") != observed_source
    ):
        raise ValueError(
            "run/preflight source provenance differs from the current imported checkout"
        )
    index_url = _validate_endpoint_identity(
        summary.get("index_url"),
        config.retrieval.index,
    )
    if index_url != config.retrieval.endpoint_url:
        raise ValueError("run endpoint differs from the single approved HTTPS URL")

    for directory in (
        ledger_dir / "attempts",
        ledger_dir / "raw",
        ledger_dir / "candidates",
        ledger_dir / "outcomes",
        ledger_dir / "cache_hits",
    ):
        if not directory.is_dir():
            raise ValueError("frozen ledger directory structure is incomplete")
    ledger = RetrievalLedger(
        ledger_dir,
        shared_cache_dir=None,
        max_calls=config.cost.max_external_requests,
        max_calls_per_topic=config.cost.max_unique_requests_per_topic,
        min_results=config.retrieval.min_results,
        required_text_results=50,
    )
    report = ledger.validate_run()
    report_payload = {
        **asdict(report),
        "external_calls": report.external_calls,
        "planned_requests": report.planned_requests,
    }
    stored_report = _read_json_object(
        required["ledger_validation"],
        "ledger validation",
    )
    if stored_report != report_payload or summary.get("ledger_validation") != report_payload:
        raise ValueError("ledger validation evidence is not freshly derived")

    raw_registry = json.loads(required["query_registry"].read_text(encoding="utf-8"))
    if not isinstance(raw_registry, list):
        raise ValueError("query registry must be a JSON array")
    base_variants = [variant for plan in plans for variant in plan.query_variants()]
    expected_base_registry = _canonical_registry_rows(base_variants, stage="base")
    base_count = len(expected_base_registry)
    if raw_registry[:base_count] != expected_base_registry:
        raise ValueError("execution base registry differs from replayed plans")

    requests_by_query: dict[tuple[str, str], object] = {}
    requests_by_key: dict[str, object] = {}
    for row in raw_registry:
        if not isinstance(row, dict):
            raise ValueError("query registry row must be an object")
        try:
            query = QueryVariant(
                topic_id=row["topic_id"],
                variant_name=row["canonical_variant_name"],
                query_text=row["query_text"],
                source_type="det_sparse_semantic_replay",
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("query registry row identity is incomplete") from exc
        request = retrieval_request_for_query(config, query, index_url=index_url)
        key = (query.topic_id, query.query_text)
        if key in requests_by_query or request.identity.request_key in requests_by_key:
            raise ValueError("query registry contains duplicate exact request identity")
        requests_by_query[key] = request
        requests_by_key[request.identity.request_key] = request
    reservation_paths = sorted((ledger_dir / "attempts").glob("*.reservation.json"))
    if len(reservation_paths) != len(requests_by_key):
        raise ValueError("query registry and ledger reservation counts differ")
    reservation_keys = {
        path.name.removesuffix(".reservation.json") for path in reservation_paths
    }
    if reservation_keys != set(requests_by_key):
        raise ValueError("query registry request keys differ from ledger reservations")

    results_by_query = {}
    for key, request in requests_by_key.items():
        reservation = _read_json_object(
            ledger_dir / "attempts" / f"{key}.reservation.json",
            "ledger reservation",
        )
        if (
            reservation.get("identity") != request.identity.canonical_dict()
            or reservation.get("query_text") != request.query_text
        ):
            raise ValueError("ledger reservation differs from canonical registry request")
        result = ledger.load_verified_result(request)
        if len(result.candidates) != 100:
            raise ValueError("formal ledger result does not contain exactly 100 candidates")
        results_by_query[(request.identity.topic_id, request.query_text)] = result

    expected_budget_root = global_budget_dir(config.root_dir).resolve()
    if summary.get("global_budget_root") != str(expected_budget_root):
        raise ValueError("run summary global budget root is not the shared fixed root")
    raw_receipts = _read_json_array(
        required["global_budget_receipts"],
        "global budget receipts",
    )
    if any(not isinstance(row, dict) for row in raw_receipts):
        raise ValueError("global budget receipt row is not an object")
    receipts = [dict(row) for row in raw_receipts]
    if not (expected_budget_root / "budget.json").is_file() or not (
        expected_budget_root / "tickets"
    ).is_dir():
        raise ValueError("durable global budget state is missing")
    global_budget = GlobalExternalBudget(expected_budget_root)
    global_budget.verify_receipts(receipts, requests_by_key)
    if summary.get("global_budget_receipts") != len(receipts):
        raise ValueError("run summary global budget receipt count differs")

    expected_prf = {}
    expansions_by_topic = {}
    expanded_variants = []
    for plan in plans:
        variants = plan.query_variants()
        eligible = (variants[0], *variants[2:5])
        existing_texts = [variant.query_text for variant in variants]
        per_text = {}
        topic_expansions = {}
        for logical in eligible:
            if logical.query_text not in per_text:
                request = requests_by_query[(plan.topic_id, logical.query_text)]
                per_text[logical.query_text] = build_verified_prf_from_ledger(
                    ledger,
                    request,
                    query_analyzer=query_analyzer,
                    existing_query_texts=existing_texts,
                )
            expansion = per_text[logical.query_text]
            topic_expansions[logical.variant_name] = expansion
            expanded = expansion.query_variant(
                variant_name=f"{logical.variant_name}:prf"
            )
            if expanded is not None:
                expanded_variants.append(expanded)
                existing_texts.append(expanded.query_text)
        expansions_by_topic[plan.topic_id] = topic_expansions
        expected_prf[plan.topic_id] = {
            name: expansion.to_dict()
            for name, expansion in topic_expansions.items()
        }
    actual_prf = _read_json_object(required["prf_artifacts"], "PRF artifacts")
    if actual_prf != _json_normalize(expected_prf):
        raise ValueError("PRF artifacts do not replay from verified ledger evidence")

    expected_registry = [
        *expected_base_registry,
        *_canonical_registry_rows(expanded_variants, stage="expanded"),
    ]
    if raw_registry != expected_registry:
        raise ValueError("query registry differs from replayed base/expanded queries")
    registry_counts = {
        topic_id: sum(row["topic_id"] == topic_id for row in expected_registry)
        for topic_id in PILOT_TOPIC_IDS
    }
    if (
        registry_counts != report.per_topic_external_calls
        or any(value > 9 for value in registry_counts.values())
        or sum(registry_counts.values()) > 36
    ):
        raise ValueError("replayed registry violates ledger or cost ceilings")

    arm_plans = {
        plan.topic_id: build_arm_plans(
            plan,
            expansions=expansions_by_topic[plan.topic_id],
        )
        for plan in plans
    }
    expected_arm_artifact = {
        topic_id: [arm.to_dict() for arm in rows]
        for topic_id, rows in arm_plans.items()
    }
    actual_arms = _read_json_object(required["arm_plans"], "arm plans")
    if actual_arms != _json_normalize(expected_arm_artifact) or any(
        arm.status != "ok" or arm.failure is not None
        for rows in arm_plans.values()
        for arm in rows
    ):
        raise ValueError("arm plans do not replay exactly or contain fallback")

    frozen_rankings = _load_frozen_ranking_files(required)
    expected_rankings: dict[str, list[RankedCandidate]] = {
        arm: [] for arm in ARM_NAMES
    }
    for topic_id in PILOT_TOPIC_IDS:
        for arm in arm_plans[topic_id]:
            pool: list[RetrievedCandidate] = []
            for stream in arm.streams:
                request = requests_by_query[(topic_id, stream.query.query_text)]
                result = results_by_query[(topic_id, stream.query.query_text)]
                pool.extend(
                    RetrievedCandidate(
                        topic_id=topic_id,
                        variant_name=stream.query.variant_name,
                        retriever_name=config.retrieval.type,
                        query_text=stream.query.query_text,
                        docid=candidate.docid,
                        rank=candidate.rank,
                        score=candidate.score,
                        text=candidate.text,
                    )
                    for candidate in result.candidates
                )
            expected_rankings[arm.arm].extend(
                fuse_arm(
                    arm,
                    pool,
                    retriever_name=config.retrieval.type,
                )
            )
    if {
        arm: [asdict(row) for row in rows]
        for arm, rows in frozen_rankings.items()
    } != {
        arm: [asdict(row) for row in rows]
        for arm, rows in expected_rankings.items()
    }:
        raise ValueError("frozen rankings do not replay from ledger and arm plans")
    actual_hashes = rankings_sha256(frozen_rankings)
    if metadata.get("ranking_sha256") != actual_hashes:
        raise ValueError("ranking artifact hashes differ from final freeze metadata")

    def overlap(left: set[str], right: set[str]) -> dict[str, object]:
        union = left | right
        return {
            "unique_vs_original_at_100": len(left - right),
            "jaccard_with_original_at_100": (
                len(left & right) / len(union) if union else 1.0
            ),
        }

    original_docs = {
        plan.topic_id: {
            candidate.docid
            for candidate in results_by_query[
                (plan.topic_id, plan.query_variants()[0].query_text)
            ].candidates
        }
        for plan in plans
    }
    logical_streams: dict[str, list[dict[str, object]]] = {
        topic_id: [] for topic_id in PILOT_TOPIC_IDS
    }
    for row in expected_registry:
        topic_id = str(row["topic_id"])
        result = results_by_query[(topic_id, str(row["query_text"]))]
        documents = {candidate.docid for candidate in result.candidates}
        for alias in row["logical_variant_aliases"]:
            logical_streams[topic_id].append(
                {
                    "stage": row["stage"],
                    "canonical_query_id": row["canonical_query_id"],
                    "logical_variant_name": alias,
                    "request_key": result.request_key,
                    **overlap(documents, original_docs[topic_id]),
                }
            )

    arm_overlap: dict[str, dict[str, dict[str, object]]] = {}
    for arm in ARM_NAMES:
        arm_overlap[arm] = {}
        for topic_id in PILOT_TOPIC_IDS:
            documents = {
                row.docid for row in frozen_rankings[arm] if row.topic_id == topic_id
            }
            baseline = {
                row.docid
                for row in frozen_rankings["O"]
                if row.topic_id == topic_id
            }
            arm_overlap[arm][topic_id] = overlap(documents, baseline)

    elapsed_by_topic: dict[str, list[float]] = {
        topic_id: [] for topic_id in PILOT_TOPIC_IDS
    }
    for key, request in requests_by_key.items():
        raw_metadata = _read_json_object(
            ledger_dir / "raw" / f"{key}.metadata.json",
            "raw retrieval metadata",
        )
        elapsed = raw_metadata.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ValueError("verified raw retrieval latency is invalid")
        elapsed_by_topic[request.identity.topic_id].append(float(elapsed))

    return {
        "schema_version": "det_sparse_descriptive_diagnostics_v1",
        "logical_stream_overlap": logical_streams,
        "arm_overlap": arm_overlap,
        "cost": {
            "external_calls": report.external_calls,
            "cache_hits": report.cache_hits,
            "per_topic_external_calls": report.per_topic_external_calls,
            "elapsed_seconds_total": math.fsum(
                elapsed
                for values in elapsed_by_topic.values()
                for elapsed in values
            ),
            "per_topic_elapsed_seconds": {
                topic_id: math.fsum(elapsed_by_topic[topic_id])
                for topic_id in PILOT_TOPIC_IDS
            },
        },
    }

def validate_evaluation_freeze(
    manifest_path: Path,
    *,
    artifact_root: Path,
    expected_qrels_path: Path,
    query_analyzer: QueryAnalyzer | None = None,
) -> dict[str, object]:
    manifest = verify_freeze_manifest(
        manifest_path,
        artifact_root=artifact_root,
        expected_qrels_path=expected_qrels_path,
    )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("final freeze lacks metadata")
    _validate_final_metadata(metadata)
    frozen_paths = {
        row["path"]
        for row in manifest["artifacts"]  # type: ignore[index]
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    required = metadata["required_artifacts"]
    assert isinstance(required, Mapping)
    missing = sorted(set(required.values()) - frozen_paths)
    if missing:
        raise ValueError("final freeze omits required artifacts: " + ", ".join(missing))
    if query_analyzer is None:
        raise ValueError("final semantic validation requires the frozen local analyzer")
    diagnostics = _validate_final_artifact_semantics(
        manifest,
        artifact_root=artifact_root,
        expected_qrels_path=expected_qrels_path,
        query_analyzer=query_analyzer,
    )
    # Re-hash everything after semantic reads to close ordinary TOCTOU drift.
    verify_freeze_manifest(
        manifest_path,
        artifact_root=artifact_root,
        expected_qrels_path=expected_qrels_path,
    )
    result = dict(manifest)
    result["derived_diagnostics"] = diagnostics
    return result


def derive_frozen_ungraded_diagnostics(
    *,
    manifest_path: Path,
    artifact_root: Path,
    expected_qrels_path: Path,
    query_analyzer: QueryAnalyzer | None = None,
) -> dict[str, object]:
    """Return qrels-free overlap, novelty, call, cache, and latency evidence."""

    validated = validate_evaluation_freeze(
        manifest_path,
        artifact_root=artifact_root,
        expected_qrels_path=expected_qrels_path,
        query_analyzer=query_analyzer,
    )
    diagnostics = validated.get("derived_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("validated freeze lacks derived diagnostics")
    return diagnostics


@dataclass(frozen=True)
class FrozenEvaluationResult:
    arm_metrics: dict[str, dict[str, object]]
    evidence: dict[str, object]
    descriptive_diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_arm_metrics(
    arm_metrics: Mapping[str, Mapping[str, object]],
) -> None:
    if tuple(arm_metrics) != ARM_NAMES:
        raise ValueError(f"arm metrics must be ordered as {ARM_NAMES!r}")
    for arm in ARM_NAMES:
        payload = arm_metrics[arm]
        if set(payload) != {"metrics", "per_topic"}:
            raise ValueError(f"{arm} metrics fields differ from the frozen schema")
        aggregate = payload.get("metrics")
        per_topic = payload.get("per_topic")
        if not isinstance(aggregate, Mapping) or tuple(aggregate) != EVALUATION_METRICS:
            raise ValueError(f"{arm} aggregate metrics are incomplete or reordered")
        if not isinstance(per_topic, Mapping) or tuple(per_topic) != PILOT_TOPIC_IDS:
            raise ValueError(f"{arm} per-topic metrics lack exact pilot coverage")
        rows = [aggregate, *(per_topic[topic_id] for topic_id in PILOT_TOPIC_IDS)]
        for row in rows:
            if not isinstance(row, Mapping) or tuple(row) != EVALUATION_METRICS:
                raise ValueError(f"{arm} metric row differs from the frozen schema")
            for value in row.values():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError(f"{arm} metric value is invalid")


def evaluate_frozen_arms(
    *,
    manifest_path: Path,
    artifact_root: Path,
    qrels_path: Path,
    rankings: Mapping[str, list[RankedCandidate]],
    query_analyzer: QueryAnalyzer | None = None,
    metrics: Sequence[str] = EVALUATION_METRICS,
    relevance_threshold: int = 2,
) -> FrozenEvaluationResult:
    """Verify the complete freeze before the first qrels read."""

    validated = validate_evaluation_freeze(
        manifest_path,
        artifact_root=artifact_root,
        expected_qrels_path=qrels_path,
        query_analyzer=query_analyzer,
    )
    metadata = validated.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("freeze manifest lacks evaluation metadata")
    expected_ranking_hashes = metadata.get("ranking_sha256")
    if not isinstance(expected_ranking_hashes, Mapping):
        raise ValueError("freeze manifest lacks frozen ranking hashes")
    actual_ranking_hashes = rankings_sha256(rankings)
    if dict(expected_ranking_hashes) != actual_ranking_hashes:
        raise ValueError("in-memory rankings differ from the frozen artifacts")
    if tuple(metrics) != EVALUATION_METRICS:
        raise ValueError("evaluation metrics differ from the frozen protocol")
    if relevance_threshold != 2:
        raise ValueError("relevance threshold differs from the frozen protocol")

    # This is intentionally the first operation that reads qrels content.  One
    # exact byte snapshot is both hashed and parsed, avoiding a hash/evaluation
    # time-of-check/time-of-use split.
    qrels_bytes = qrels_path.read_bytes()
    qrels_sha256 = hashlib.sha256(qrels_bytes).hexdigest()
    qrels = parse_qrels_bytes(qrels_bytes)
    arm_metrics = {
        arm: evaluate_ranked(
            rankings[arm],
            qrels,
            metric_names=metrics,
            relevance_threshold=relevance_threshold,
        )
        for arm in ARM_NAMES
    }
    _validate_arm_metrics(arm_metrics)
    ungraded = validated.get("derived_diagnostics")
    if not isinstance(ungraded, dict):
        raise ValueError("validated freeze lacks qrels-free diagnostics")
    baseline_docs = {
        topic_id: {
            row.docid for row in rankings["O"] if row.topic_id == topic_id
        }
        for topic_id in PILOT_TOPIC_IDS
    }
    graded_novelty = {
        arm: {
            topic_id: {
                "new_relevant_vs_original_at_100": len(new_relevant),
                "new_relevant_docids": sorted(new_relevant),
            }
            for topic_id in PILOT_TOPIC_IDS
            for new_relevant in [
                (
                    {
                        row.docid
                        for row in rankings[arm]
                        if row.topic_id == topic_id
                    }
                    - baseline_docs[topic_id]
                )
                & {
                    docid
                    for docid, grade in qrels.get(topic_id, {}).items()
                    if grade >= relevance_threshold
                }
            ]
        }
        for arm in ARM_NAMES
    }
    descriptive_diagnostics = {
        **ungraded,
        "graded_new_relevant_vs_original": graded_novelty,
    }
    descriptive_hash = hashlib.sha256(
        _canonical_json(descriptive_diagnostics)
    ).hexdigest()
    return FrozenEvaluationResult(
        arm_metrics=arm_metrics,
        evidence={
            "schema_version": "det_sparse_frozen_evaluation_v1",
            "experiment_id": EXPERIMENT_ID,
            "topic_ids": list(PILOT_TOPIC_IDS),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "qrels_path": str(qrels_path.resolve()),
            "qrels_sha256": qrels_sha256,
            "ranking_sha256": actual_ranking_hashes,
            "metrics": list(EVALUATION_METRICS),
            "relevance_threshold": relevance_threshold,
            "descriptive_diagnostics_sha256": descriptive_hash,
        },
        descriptive_diagnostics=descriptive_diagnostics,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


@dataclass(frozen=True)
class ComponentGate:
    passed: bool
    mean_recall_delta: float
    nonnegative_topics: int
    mean_ndcg_delta: float
    worst_ndcg_delta: float


@dataclass(frozen=True)
class PilotDecision:
    mechanical_passed: bool
    facets: ComponentGate
    expansion: ComponentGate
    combined: ComponentGate
    combined_mean_delta_vs_facets: float
    combined_mean_delta_vs_expansion: float
    preferred_arm: str
    evaluation_evidence: dict[str, object]
    descriptive_diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_pilot(
    *,
    manifest_path: Path,
    artifact_root: Path,
    qrels_path: Path,
    rankings: Mapping[str, list[RankedCandidate]],
    query_analyzer: QueryAnalyzer | None = None,
) -> PilotDecision:
    """Evaluate frozen evidence and apply the preregistered paired gates."""

    evaluation = evaluate_frozen_arms(
        manifest_path=manifest_path,
        artifact_root=artifact_root,
        qrels_path=qrels_path,
        rankings=rankings,
        query_analyzer=query_analyzer,
    )
    arm_metrics = evaluation.arm_metrics
    _validate_arm_metrics(arm_metrics)
    mechanical_passed = True

    def topic_value(arm: str, topic_id: str, metric: str) -> float:
        per_topic = arm_metrics[arm].get("per_topic")
        if not isinstance(per_topic, Mapping):
            raise ValueError(f"{arm} metrics lack per_topic values")
        row = per_topic.get(topic_id)
        if not isinstance(row, Mapping) or metric not in row:
            raise ValueError(f"{arm}/{topic_id} lacks {metric}")
        return float(row[metric])

    recalls = {
        arm: [
            topic_value(arm, topic_id, "recall@100")
            for topic_id in PILOT_TOPIC_IDS
        ]
        for arm in ARM_NAMES
    }
    ndcgs = {
        arm: [topic_value(arm, topic_id, "ndcg@10") for topic_id in PILOT_TOPIC_IDS]
        for arm in ARM_NAMES
    }

    def component(arm: str, minimum_recall_delta: float) -> ComponentGate:
        recall_deltas = [
            candidate - baseline
            for candidate, baseline in zip(recalls[arm], recalls["O"])
        ]
        ndcg_deltas = [
            candidate - baseline
            for candidate, baseline in zip(ndcgs[arm], ndcgs["O"])
        ]
        mean_recall = _mean(recall_deltas)
        nonnegative = sum(delta >= 0 for delta in recall_deltas)
        mean_ndcg = _mean(ndcg_deltas)
        worst_ndcg = min(ndcg_deltas)
        passed = (
            mechanical_passed
            and mean_recall >= minimum_recall_delta
            and nonnegative >= 3
            and mean_ndcg >= -0.02
            and worst_ndcg >= -0.10
        )
        return ComponentGate(
            passed=passed,
            mean_recall_delta=mean_recall,
            nonnegative_topics=nonnegative,
            mean_ndcg_delta=mean_ndcg,
            worst_ndcg_delta=worst_ndcg,
        )

    facets = component("F", 0.010)
    expansion = component("E", 0.005)
    combined_base = component("FE", 0.015)
    combined_vs_facets = _mean(
        [combined - facet for combined, facet in zip(recalls["FE"], recalls["F"])]
    )
    combined_vs_expansion = _mean(
        [combined - expansion_value for combined, expansion_value in zip(recalls["FE"], recalls["E"])]
    )
    combined = ComponentGate(
        passed=(
            combined_base.passed
            and combined_vs_facets >= 0.0025
            and combined_vs_expansion >= 0.0025
        ),
        mean_recall_delta=combined_base.mean_recall_delta,
        nonnegative_topics=combined_base.nonnegative_topics,
        mean_ndcg_delta=combined_base.mean_ndcg_delta,
        worst_ndcg_delta=combined_base.worst_ndcg_delta,
    )
    if combined.passed:
        preferred = "FE"
    elif facets.passed and expansion.passed:
        preferred = "F" if facets.mean_recall_delta >= expansion.mean_recall_delta else "E"
    elif facets.passed:
        preferred = "F"
    elif expansion.passed:
        preferred = "E"
    else:
        preferred = "O"
    return PilotDecision(
        mechanical_passed=mechanical_passed,
        facets=facets,
        expansion=expansion,
        combined=combined,
        combined_mean_delta_vs_facets=combined_vs_facets,
        combined_mean_delta_vs_expansion=combined_vs_expansion,
        preferred_arm=preferred,
        evaluation_evidence=evaluation.evidence,
        descriptive_diagnostics=evaluation.descriptive_diagnostics,
    )
