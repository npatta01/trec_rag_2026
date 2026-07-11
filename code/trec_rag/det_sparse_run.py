"""End-to-end, injected-transport executor for the frozen sparse pilot.

This module intentionally provides no HTTP implementation or CLI.  Entering an
external service remains a separate authorization boundary supplied by the
caller.  All planning, identity checks, budgets, raw-first evidence, PRF,
fusion, and final freezing are bound here so an ad-hoc caller cannot skip a
stage accidentally.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import unquote, urlsplit

from trec_rag.det_sparse_arms import (
    ArmFailure,
    ArmPlan,
    ArmStream,
    StreamMembership,
    build_arm_plans,
    build_verified_prf_from_ledger,
    fuse_arm,
    open_frozen_retrieval_ledger,
    projected_unique_request_ceiling,
    retrieval_request_for_query,
)
from trec_rag.det_sparse_config import (
    ARM_NAMES,
    DetSparseConfig,
    load_det_sparse_config,
)
from trec_rag.det_sparse_budget import (
    GlobalExternalBudget,
    GloballyBudgetedTransport,
    global_budget_dir,
)
from trec_rag.det_sparse_freeze import (
    FINAL_ARTIFACT_KEYS,
    create_evaluation_freeze,
    validate_evaluation_freeze,
)
from trec_rag.det_sparse_ledger import (
    RetrievalLedger,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTransport,
)
from trec_rag.det_sparse_preflight import validate_preflight
from trec_rag.det_sparse_provenance import (
    current_runtime_provenance,
    current_source_provenance,
)
from trec_rag.det_sparse_transport import validate_transport_binding
from trec_rag.deterministic_sparse import (
    PLANNER_VERSION,
    DeterministicSparsePlan,
    PrfExpansion,
    build_deterministic_sparse_plan,
)
from trec_rag.pipeline_models import QueryVariant, RankedCandidate, RetrievedCandidate
from trec_rag.det_sparse_transport import FrozenHttpsRetrievalTransport
from trec_rag.query_analyzer import QueryAnalyzer, RemoteLuceneQueryAnalyzer


RUN_SCHEMA_VERSION = "det_sparse_execution_v1"
_ANALYZER_CONFORMANCE_PROBES = (
    ("City buses are running on time.", ("citi", "buse", "run", "time")),
    ("open-source banks banking bank's not how", ("open", "sourc", "bank", "bank", "bank", "how")),
    ("no not such then there these will", ()),
)


@dataclass(frozen=True)
class CanonicalQuery:
    topic_id: str
    stage: str
    canonical_query_id: str
    canonical_variant_name: str
    query_text: str
    query_sha256: str
    logical_variant_aliases: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DetSparseExecutionResult:
    status: str
    mechanical_valid: bool
    output_dir: Path
    external_calls: int
    per_topic_external_calls: dict[str, int]
    rankings: dict[str, list[RankedCandidate]]
    final_freeze_path: Path | None
    failure: dict[str, str] | None


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


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
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_json(path: Path, value: object) -> None:
    _create_only(path, _canonical_bytes(value, pretty=True))


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    content = b"".join(
        _canonical_bytes(asdict(row) if hasattr(row, "__dataclass_fields__") else row)
        for row in rows
    )
    _create_only(path, content)


def _fingerprint_sha256(query_analyzer: QueryAnalyzer) -> str:
    return hashlib.sha256(
        json.dumps(
            query_analyzer.fingerprint.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _validate_formal_runtime_boundary(
    config: DetSparseConfig,
    query_analyzer: QueryAnalyzer,
    transport: RetrievalTransport,
) -> dict[str, object]:
    """Prove the actual network/analyzer implementations before any ticket."""

    validate_transport_binding(transport, config.retrieval.endpoint_url)
    if (
        type(transport) is not FrozenHttpsRetrievalTransport
        or transport.formal_network_ready is not True
    ):
        raise ValueError("formal execution requires the built-in non-injected HTTPS transport")
    if (
        type(query_analyzer) is not RemoteLuceneQueryAnalyzer
        or query_analyzer.base_url != config.analyzer.url
        or query_analyzer.timeout != 10.0
    ):
        raise ValueError("formal execution requires the exact loopback Lucene analyzer client")
    probe_hashes = {}
    for text, expected in _ANALYZER_CONFORMANCE_PROBES:
        analyzed = query_analyzer.analyze(text)
        if analyzed.tokens != expected:
            raise ValueError("local analyzer failed the frozen conformance probes")
        probe_hashes[hashlib.sha256(text.encode("utf-8")).hexdigest()] = hashlib.sha256(
            json.dumps(list(analyzed.tokens), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return {
        "transport_version": transport.transport_version,
        "transport_class": (
            "trec_rag.det_sparse_transport.FrozenHttpsRetrievalTransport"
        ),
        "endpoint_url": transport.endpoint_url,
        "one_shot_no_retry": True,
        "redirects_allowed": False,
        "injected_opener": False,
        "analyzer_client_class": (
            "trec_rag.query_analyzer.RemoteLuceneQueryAnalyzer"
        ),
        "analyzer_url": query_analyzer.base_url,
        "analyzer_conformance_probe_sha256": probe_hashes,
    }


def _require_external_authorization(config: DetSparseConfig) -> None:
    """Hard gate that must run before touching any caller-supplied runtime."""

    if config.retrieval.index_revision == "hosted_climbmix_unknown_revision":
        raise PermissionError(
            "external retrieval remains blocked until the hosted index exposes "
            "an immutable revision and a separate authorization gate approves it"
        )


def _validate_endpoint(index_url: str, expected_index: str) -> None:
    parsed = urlsplit(index_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("retrieval endpoint URL is not an admissible exact search URL")
    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 3 or parts[-3:] != ["v1", expected_index, "search"]:
        raise ValueError("retrieval endpoint path does not encode the frozen index")


def _load_frozen_plans(
    config: DetSparseConfig,
    preflight_dir: Path,
    query_analyzer: QueryAnalyzer,
) -> tuple[DeterministicSparsePlan, ...]:
    validate_preflight(config, preflight_dir)
    metadata = json.loads((preflight_dir / "_preflight.json").read_text(encoding="utf-8"))
    plans: list[DeterministicSparsePlan] = []
    for row in metadata["plans"]:
        stored = json.loads(
            (preflight_dir / row["plan_path"]).read_text(encoding="utf-8")
        )
        rebuilt = build_deterministic_sparse_plan(
            topic_id=str(stored["topic_id"]),
            narrative=str(stored["original_query_text"]),
            query_analyzer=query_analyzer,
        )
        rebuilt_json = json.loads(
            json.dumps(rebuilt.to_dict(), ensure_ascii=False, sort_keys=True)
        )
        if rebuilt_json != stored:
            raise ValueError("frozen plan does not reproduce under the current analyzer/code")
        if rebuilt.status != "ok":
            raise ValueError("formal execution refuses a fallback preflight plan")
        plans.append(rebuilt)
    if tuple(plan.topic_id for plan in plans) != config.topic_ids:
        raise ValueError("frozen plans differ from the exact topic order")
    return tuple(plans)


def _canonicalize(
    variants: Sequence[QueryVariant],
    *,
    stage: str,
) -> tuple[CanonicalQuery, ...]:
    by_topic_text: dict[tuple[str, str], list[QueryVariant]] = {}
    order: list[tuple[str, str]] = []
    for variant in variants:
        key = (variant.topic_id, variant.query_text)
        if key not in by_topic_text:
            by_topic_text[key] = []
            order.append(key)
        by_topic_text[key].append(variant)
    ordinals: dict[str, int] = {}
    rows: list[CanonicalQuery] = []
    for topic_id, query_text in order:
        ordinals[topic_id] = ordinals.get(topic_id, 0) + 1
        aliases = by_topic_text[(topic_id, query_text)]
        rows.append(
            CanonicalQuery(
                topic_id=topic_id,
                stage=stage,
                canonical_query_id=f"{topic_id}:{stage}:{ordinals[topic_id]:02d}",
                canonical_variant_name=aliases[0].variant_name,
                query_text=query_text,
                query_sha256=hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                logical_variant_aliases=tuple(alias.variant_name for alias in aliases),
            )
        )
    return tuple(rows)


def _verify_preflight_registry(
    preflight_dir: Path,
    expected: Sequence[CanonicalQuery],
) -> None:
    stored = json.loads(
        (preflight_dir / "base_query_registry.json").read_text(encoding="utf-8")
    )
    comparable = [
        {
            "topic_id": row.topic_id,
            "canonical_query_id": row.canonical_query_id,
            "canonical_variant_name": row.canonical_variant_name,
            "query_text": row.query_text,
            "query_sha256": row.query_sha256,
            "logical_variant_aliases": list(row.logical_variant_aliases),
        }
        for row in expected
    ]
    stripped = [
        {key: value for key, value in row.items() if key != "source_types"}
        for row in stored
    ]
    if stripped != comparable:
        raise ValueError("preflight canonical query registry does not reproduce")


def _request_for_row(
    config: DetSparseConfig,
    row: CanonicalQuery,
    *,
    index_url: str,
) -> RetrievalRequest:
    return retrieval_request_for_query(
        config,
        QueryVariant(
            topic_id=row.topic_id,
            variant_name=row.canonical_variant_name,
            query_text=row.query_text,
            source_type=f"{PLANNER_VERSION}_{row.stage}",
        ),
        index_url=index_url,
    )


def _retrieve_rows(
    config: DetSparseConfig,
    ledger: RetrievalLedger,
    rows: Sequence[CanonicalQuery],
    *,
    index_url: str,
    transport: RetrievalTransport,
    results: dict[tuple[str, str], tuple[RetrievalRequest, RetrievalResult]],
) -> None:
    for row in rows:
        key = (row.topic_id, row.query_text)
        if key in results:
            raise ValueError("canonical registry attempted a duplicate exact query")
        request = _request_for_row(config, row, index_url=index_url)
        result = ledger.retrieve(request, transport)
        if len(result.candidates) != config.retrieval.hits:
            raise ValueError(
                f"formal retrieval requires exactly {config.retrieval.hits} rows; "
                f"received {len(result.candidates)} for {row.canonical_query_id}"
            )
        results[key] = (request, result)


def _logical_candidates(
    query: QueryVariant,
    result: RetrievalResult,
    *,
    retriever_name: str,
) -> list[RetrievedCandidate]:
    return [
        RetrievedCandidate(
            topic_id=query.topic_id,
            variant_name=query.variant_name,
            retriever_name=retriever_name,
            query_text=query.query_text,
            docid=row.docid,
            rank=row.rank,
            score=row.score,
            text=row.text,
        )
        for row in result.candidates
    ]


def _fallback_arms(
    plan: DeterministicSparsePlan,
    message: str,
) -> tuple[ArmPlan, ...]:
    original = plan.query_variants()[0]
    stream = ArmStream(
        query=original,
        weight=1.0,
        memberships=(StreamMembership("original", 1.0, True),),
    )
    return tuple(
        ArmPlan(
            topic_id=plan.topic_id,
            arm=arm,
            status="ok" if arm == "O" else "fallback",
            streams=(stream,),
            failure=(
                None
                if arm == "O"
                else ArmFailure("execution_failure", message)
            ),
        )
        for arm in ARM_NAMES
    )


def _freeze_source_from_preflight(preflight_dir: Path) -> dict[str, object]:
    metadata = json.loads((preflight_dir / "_preflight.json").read_text(encoding="utf-8"))
    source = metadata.get("source")
    if not isinstance(source, dict):
        raise ValueError("preflight lacks source provenance")
    return source


def _current_source_provenance(root: Path) -> dict[str, object]:
    return current_source_provenance(root)


def _runtime_provenance(root: Path) -> dict[str, object]:
    return current_runtime_provenance(root)


def execute_det_sparse_run(
    config: DetSparseConfig,
    *,
    preflight_dir: Path,
    run_dir: Path,
    query_analyzer: QueryAnalyzer,
    transport: RetrievalTransport,
) -> DetSparseExecutionResult:
    """Execute the complete qrels-blind run through a supplied transport."""

    # The dataclass is convenient for callers, but it is not an authority.  In
    # particular, ``dataclasses.replace`` must not be able to change a paid-call
    # identity (endpoint/index/depth/analyzer), the qrels firewall, or the fixed
    # output path while still pointing at an unchanged frozen YAML file.
    canonical_config = load_det_sparse_config(config.config_path)
    if config != canonical_config:
        raise ValueError(
            "executor config object differs from a fresh canonical config reload"
        )
    config = canonical_config
    index_url = config.retrieval.endpoint_url

    preflight_dir = preflight_dir.resolve()
    run_dir = run_dir.resolve()
    output_root = config.output_dir.resolve()
    expected_preflight = output_root / "preflight"
    expected_run = output_root / "execution"
    if preflight_dir != expected_preflight or run_dir != expected_run:
        raise ValueError("formal preflight/run paths are fixed and cannot reset the budget")
    try:
        preflight_dir.relative_to(output_root)
        run_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("preflight and run directories must be under experiment output") from exc
    if run_dir.exists():
        raise FileExistsError(f"execution output already exists: {run_dir}")
    _require_external_authorization(config)
    _validate_endpoint(index_url, config.retrieval.index)
    if _fingerprint_sha256(query_analyzer) != config.analyzer.expected_fingerprint_sha256:
        raise ValueError("execution analyzer fingerprint differs from frozen config")
    plans = _load_frozen_plans(config, preflight_dir, query_analyzer)
    frozen_source = _freeze_source_from_preflight(preflight_dir)
    current_source = _current_source_provenance(config.root_dir)
    if current_source != frozen_source:
        raise ValueError("current source commit/tree differs from frozen preflight")
    base_variants = tuple(
        variant for plan in plans for variant in plan.query_variants()
    )
    base_registry = _canonicalize(base_variants, stage="base")
    _verify_preflight_registry(preflight_dir, base_registry)
    preflight_metadata = json.loads(
        (preflight_dir / "_preflight.json").read_text(encoding="utf-8")
    )
    raw_projections = preflight_metadata["request_projections"]
    if not isinstance(raw_projections, list):
        raise ValueError("preflight request projections are not a list")
    expected_projection_rows = [
        {
            "topic_id": plan.topic_id,
            "base_unique_requests": len(
                {query.query_text for query in plan.query_variants()}
            ),
            "derived_max_unique_requests": projected_unique_request_ceiling(plan),
        }
        for plan in plans
    ]
    if raw_projections != expected_projection_rows:
        raise ValueError("preflight request projections do not replay exactly from plans")
    projections = {
        row["topic_id"]: row["derived_max_unique_requests"]
        for row in expected_projection_rows
    }

    runtime_boundary = _validate_formal_runtime_boundary(
        config,
        query_analyzer,
        transport,
    )

    budget_root = global_budget_dir(config.root_dir)
    global_budget = GlobalExternalBudget(budget_root)
    budgeted_transport = GloballyBudgetedTransport(global_budget, transport)

    run_dir.mkdir(parents=True)
    ledger = open_frozen_retrieval_ledger(config, run_dir=run_dir / "ledger")
    results: dict[tuple[str, str], tuple[RetrievalRequest, RetrievalResult]] = {}
    rankings: dict[str, list[RankedCandidate]] = {arm: [] for arm in ARM_NAMES}
    expansions_by_topic: dict[str, dict[str, PrfExpansion]] = {}
    arm_plans_by_topic: dict[str, tuple[ArmPlan, ...]] = {}
    all_registry: list[CanonicalQuery] = list(base_registry)
    failure: dict[str, str] | None = None
    try:
        _retrieve_rows(
            config,
            ledger,
            base_registry,
            index_url=index_url,
            transport=budgeted_transport,
            results=results,
        )
        if _fingerprint_sha256(query_analyzer) != config.analyzer.expected_fingerprint_sha256:
            raise ValueError("execution analyzer fingerprint changed after base retrieval")

        expanded_variants: list[QueryVariant] = []
        for plan in plans:
            variants = plan.query_variants()
            eligible = (variants[0], *variants[2:5])
            per_text: dict[str, PrfExpansion] = {}
            topic_expansions: dict[str, PrfExpansion] = {}
            existing_texts = [variant.query_text for variant in variants]
            for logical in eligible:
                if logical.query_text not in per_text:
                    request, _result = results[(plan.topic_id, logical.query_text)]
                    per_text[logical.query_text] = build_verified_prf_from_ledger(
                        ledger,
                        request,
                        query_analyzer=query_analyzer,
                        existing_query_texts=existing_texts,
                    )
                expansion = per_text[logical.query_text]
                topic_expansions[logical.variant_name] = expansion
                if expansion.status == "failure":
                    detail = expansion.failure.message if expansion.failure else "unknown"
                    raise ValueError(f"PRF failed for {logical.variant_name}: {detail}")
                variant = expansion.query_variant(
                    variant_name=f"{logical.variant_name}:prf"
                )
                if variant is not None:
                    expanded_variants.append(variant)
                    existing_texts.append(variant.query_text)
            expansions_by_topic[plan.topic_id] = topic_expansions

        expanded_registry = _canonicalize(expanded_variants, stage="expanded")
        all_registry.extend(expanded_registry)
        registry_counts = {
            topic_id: sum(row.topic_id == topic_id for row in all_registry)
            for topic_id in config.topic_ids
        }
        if any(
            count > projections[topic_id] or count > 9
            for topic_id, count in registry_counts.items()
        ) or sum(registry_counts.values()) > 36:
            raise ValueError("canonical execution registry exceeds frozen request ceilings")
        _write_json(
            run_dir / "query_registry.json",
            [row.to_dict() for row in all_registry],
        )
        _write_json(
            run_dir / "prf_artifacts.json",
            {
                topic_id: {
                    variant_name: expansion.to_dict()
                    for variant_name, expansion in rows.items()
                }
                for topic_id, rows in expansions_by_topic.items()
            },
        )
        _retrieve_rows(
            config,
            ledger,
            expanded_registry,
            index_url=index_url,
            transport=budgeted_transport,
            results=results,
        )
        if _fingerprint_sha256(query_analyzer) != config.analyzer.expected_fingerprint_sha256:
            raise ValueError("execution analyzer fingerprint changed after PRF retrieval")

        for plan in plans:
            arm_plans = build_arm_plans(
                plan,
                expansions=expansions_by_topic[plan.topic_id],
            )
            if any(arm.status != "ok" for arm in arm_plans):
                raise ValueError("successful execution produced an unexpected arm fallback")
            arm_plans_by_topic[plan.topic_id] = arm_plans
            for arm in arm_plans:
                pool: list[RetrievedCandidate] = []
                for stream in arm.streams:
                    _request, result = results[(plan.topic_id, stream.query.query_text)]
                    pool.extend(
                        _logical_candidates(
                            stream.query,
                            result,
                            retriever_name=config.retrieval.type,
                        )
                    )
                rankings[arm.arm].extend(
                    fuse_arm(
                        arm,
                        pool,
                        retriever_name=config.retrieval.type,
                    )
                )
        if _current_source_provenance(config.root_dir) != frozen_source:
            raise ValueError("source commit/tree changed during execution")
        if hashlib.sha256(config.config_path.read_bytes()).hexdigest() != str(
            preflight_metadata["config_sha256"]
        ):
            raise ValueError("frozen config changed during execution")
    except Exception as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc)}
        rankings = {arm: [] for arm in ARM_NAMES}
        arm_plans_by_topic = {}
        for plan in plans:
            fallback_arms = _fallback_arms(plan, str(exc))
            arm_plans_by_topic[plan.topic_id] = fallback_arms
            original = plan.query_variants()[0]
            stored = results.get((plan.topic_id, original.query_text))
            if stored is None:
                continue
            _request, result = stored
            pool = _logical_candidates(
                original,
                result,
                retriever_name=config.retrieval.type,
            )
            for arm in fallback_arms:
                rankings[arm.arm].extend(
                    fuse_arm(
                        arm,
                        pool,
                        retriever_name=config.retrieval.type,
                    )
                )

    # Every terminal state is durable and qrels-blind.
    if not (run_dir / "query_registry.json").exists():
        _write_json(
            run_dir / "query_registry.json",
            [row.to_dict() for row in all_registry],
        )
    if not (run_dir / "prf_artifacts.json").exists():
        _write_json(
            run_dir / "prf_artifacts.json",
            {
                topic_id: {
                    variant_name: expansion.to_dict()
                    for variant_name, expansion in rows.items()
                }
                for topic_id, rows in expansions_by_topic.items()
            },
        )
    _write_json(
        run_dir / "arm_plans.json",
        {
            topic_id: [arm.to_dict() for arm in rows]
            for topic_id, rows in arm_plans_by_topic.items()
        },
    )
    for arm in ARM_NAMES:
        _write_jsonl(run_dir / "rankings" / f"{arm}.jsonl", rankings[arm])

    validation = ledger.validate_run()
    validation_payload = {
        **asdict(validation),
        "external_calls": validation.external_calls,
        "planned_requests": validation.planned_requests,
    }
    _write_json(run_dir / "ledger_validation.json", validation_payload)
    _write_json(run_dir / "global_budget_receipts.json", budgeted_transport.receipts)
    runtime = _runtime_provenance(config.root_dir)
    _write_json(run_dir / "runtime.json", runtime)
    _create_only(run_dir / "config.yaml", config.config_path.read_bytes())

    mechanical_valid = (
        failure is None
        and validation.failures == 0
        and validation.pending == 0
        and validation.successes == len(all_registry)
        and validation.cache_hits == 0
        and len(budgeted_transport.receipts) == validation.external_calls
        and all(
            len([row for row in rankings[arm] if row.topic_id == topic_id]) > 0
            for arm in ARM_NAMES
            for topic_id in config.topic_ids
        )
    )
    source = current_source
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "topic_ids": list(config.topic_ids),
        "arms": list(ARM_NAMES),
        "mechanical_valid": mechanical_valid,
        "retrieval_complete": mechanical_valid,
        "fallback_topic_ids": (
            [] if mechanical_valid else list(config.topic_ids)
        ),
        "qrels_opened": False,
        "model_calls": 0,
        "reranker_calls": 0,
        "hits": config.retrieval.hits,
        "index_id": config.retrieval.index,
        "index_url": index_url,
        "endpoint_path_index_verified": True,
        "index_revision": "hosted_climbmix_unknown_revision",
        "analyzer_fingerprint_sha256": config.analyzer.expected_fingerprint_sha256,
        "cache_scope": config.retrieval.cache_policy,
        "cache_hits": validation.cache_hits,
        "per_topic_external_calls": {
            topic_id: validation.per_topic_external_calls.get(topic_id, 0)
            for topic_id in config.topic_ids
        },
        "external_calls": validation.external_calls,
        "planned_requests": validation.planned_requests,
        "ledger_validation": validation_payload,
        "global_budget_root": str(budget_root.resolve()),
        "global_budget_receipts": len(budgeted_transport.receipts),
        "source": source,
        "runtime": runtime,
        "runtime_boundary": runtime_boundary,
        "failure": failure,
    }
    _write_json(run_dir / "run_summary.json", summary)

    freeze_path: Path | None = None
    if mechanical_valid:
        required = {
            "config": run_dir / "config.yaml",
            "preflight_freeze": preflight_dir / "pre_retrieval_freeze.json",
            "query_registry": run_dir / "query_registry.json",
            "ledger_manifest": run_dir / "ledger" / "ledger.json",
            "ledger_validation": run_dir / "ledger_validation.json",
            "global_budget_receipts": run_dir / "global_budget_receipts.json",
            "prf_artifacts": run_dir / "prf_artifacts.json",
            "arm_plans": run_dir / "arm_plans.json",
            "runtime": run_dir / "runtime.json",
            "run_summary": run_dir / "run_summary.json",
            "ranking_O": run_dir / "rankings" / "O.jsonl",
            "ranking_F": run_dir / "rankings" / "F.jsonl",
            "ranking_E": run_dir / "rankings" / "E.jsonl",
            "ranking_FE": run_dir / "rankings" / "FE.jsonl",
        }
        if tuple(required) != FINAL_ARTIFACT_KEYS:
            raise AssertionError("required artifact order drift")
        required_paths = {path.resolve() for path in required.values()}
        additional = sorted(
            (
                path
                for root in (preflight_dir, run_dir / "ledger")
                for path in root.rglob("*")
                if path.is_file()
                and path.resolve() not in required_paths
                and not path.name.endswith(".lock")
            ),
            key=lambda path: str(path),
        )
        freeze_path = run_dir / "evaluation_freeze.json"
        create_evaluation_freeze(
            freeze_path,
            artifact_root=output_root,
            required_artifacts=required,
            additional_artifacts=additional,
            qrels_path=config.evaluation.qrels,
            rankings=rankings,
            run_summary=summary,
        )
        validate_evaluation_freeze(
            freeze_path,
            artifact_root=output_root,
            expected_qrels_path=config.evaluation.qrels,
            query_analyzer=query_analyzer,
        )

    return DetSparseExecutionResult(
        status="success" if mechanical_valid else "failure",
        mechanical_valid=mechanical_valid,
        output_dir=run_dir,
        external_calls=validation.external_calls,
        per_topic_external_calls={
            topic_id: validation.per_topic_external_calls.get(topic_id, 0)
            for topic_id in config.topic_ids
        },
        rankings=rankings,
        final_freeze_path=freeze_path,
        failure=failure,
    )
