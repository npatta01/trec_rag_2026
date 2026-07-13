"""Build the canonical portable report for the facet-local MiniLM pilot.

Only already-saved Task 1--6 artifacts are accepted.  This module does not
open qrels, retrieval ledgers, candidate snapshots, review packets, ranking
rows, or inference caches.  Its optional HTML path invokes the pinned portable
renderer into a private temporary file and publishes it create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Mapping, Sequence


TITLE = "Facet-local MiniLM Pilot: Better Facet Evidence, Blocked by Fusion"
REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_TOPICS = ("200", "225", "707", "897")
PILOT_TOPIC_SET = frozenset(PILOT_TOPICS)
EVALUATION_KEYS = (
    "raw_union",
    "prefusion",
    "facet_retention",
    "systems",
    "gains_losses",
    "review_metrics",
    "representatives",
    "decision",
)
SOURCE_PATHS = {
    "manifest": "reports/experiments/facet_local_minilm_pilot_v1/manifest.json",
    "preflight": "outputs/rag25_facet_local_minilm_v1/preflight_v2/preflight.json",
    "scoring_receipt": "outputs/rag25_facet_local_minilm_v1/full_scoring_v1/scoring_receipt.json",
    "benchmark": "outputs/rag25_facet_local_minilm_v1/benchmark_v1/benchmark_telemetry.json",
    "model_download_approval": "outputs/rag25_facet_local_minilm_v1/approvals/model_download_v1.json",
    "benchmark_approval": "outputs/rag25_facet_local_minilm_v1/approvals/benchmark_v1.json",
    "full_scoring_approval": "outputs/rag25_facet_local_minilm_v1/approvals/full_scoring_v1.json",
    "ranking_freeze": "outputs/rag25_facet_local_minilm_v1/freeze_v1/freeze.json",
    "legacy_corrected_diff": "outputs/rag25_facet_local_minilm_v1/freeze_v1/legacy_corrected_diff.json",
    "review_freeze": "outputs/rag25_facet_local_minilm_v1/review_v3/review_freeze.json",
    "review_create_receipt": "outputs/rag25_facet_local_minilm_v1/review_v3/create_receipt.json",
    "raw_union": "outputs/rag25_facet_local_minilm_v1/evaluation_v1/raw_union.json",
    "prefusion": "outputs/rag25_facet_local_minilm_v1/evaluation_v1/prefusion.json",
    "facet_retention": "outputs/rag25_facet_local_minilm_v1/evaluation_v1/facet_retention.json",
    "systems": "outputs/rag25_facet_local_minilm_v1/evaluation_v1/systems.json",
    "gains_losses": "outputs/rag25_facet_local_minilm_v1/evaluation_v1/gains_losses.json",
    "review_metrics": "outputs/rag25_facet_local_minilm_v1/evaluation_v1/review_metrics.json",
    "representatives": "outputs/rag25_facet_local_minilm_v1/evaluation_v1/representatives.json",
    "representative_provenance": "outputs/rag25_facet_local_minilm_v1/derived_v2/representative_provenance_v2.json",
    "decision": "outputs/rag25_facet_local_minilm_v1/evaluation_v1/decision.json",
    "qrels_access_approval": "outputs/rag25_facet_local_minilm_v1/approvals/qrels_access_v1.json",
    "qrels_consumption_registry": "outputs/rag25_facet_local_minilm_v1/approvals/qrels_consumption_registry_v2.json",
    "qrels_consumption_legacy_marker": "outputs/rag25_facet_local_minilm_v1/approvals/qrels_access_v1.json.consumed.json",
    "qrels_access_receipt": "outputs/rag25_facet_local_minilm_v1/evaluation_v1/qrels_access_receipt.json",
}
SOURCE_LABELS = {
    "manifest": "Frozen 31-stream source manifest",
    "preflight": "Authenticated tokenizer-only preflight",
    "scoring_receipt": "Completed local MiniLM scoring receipt",
    "benchmark": "ROCm benchmark telemetry",
    "model_download_approval": "Pinned safe-file model download approval",
    "benchmark_approval": "Bounded benchmark approval",
    "full_scoring_approval": "Full local scoring approval",
    "ranking_freeze": "Pre-qrels ranking freeze",
    "legacy_corrected_diff": "Legacy versus corrected fusion audit",
    "review_freeze": "Pre-qrels blinded-review freeze",
    "review_create_receipt": "Blinded-review pool receipt",
    "raw_union": "Relevant candidate-union curves",
    "prefusion": "Pre-fusion promotion evidence",
    "facet_retention": "Per-facet retention evidence",
    "systems": "System effectiveness evaluation",
    "gains_losses": "Relevant-document gain/loss evaluation",
    "review_metrics": "Blinded facet-review metrics",
    "representatives": "Bounded representative passages",
    "representative_provenance": "Facet-event representative provenance",
    "decision": "Mechanical experiment decision",
    "qrels_access_approval": "One-time qrels-access approval",
    "qrels_consumption_registry": "Trusted path-independent qrels-consumption registry mirror",
    "qrels_consumption_legacy_marker": "Legacy approval-adjacent consumption marker",
    "qrels_access_receipt": "One-time qrels-access receipt",
}
MATERIAL_SOURCE_IDS = {
    "decision": (
        "decision",
        "raw_union",
        "prefusion",
        "systems",
        "legacy_corrected_diff",
        "review_metrics",
    ),
    "qrels_access_receipt": (
        "qrels_access_receipt",
        "qrels_access_approval",
        "qrels_consumption_registry",
        "ranking_freeze",
        "review_freeze",
        "review_create_receipt",
    ),
    "representative_provenance": (
        "representative_provenance",
        "representatives",
        "prefusion",
        "ranking_freeze",
    ),
}
SOURCE_METRIC_DEFINITIONS = {
    "manifest": [
        "One query-stream row per manifest.streams item; narrative rows are the family='original' subset.",
        "Candidate rows equal expected_rows and must sum to 3,100 across exactly 31 streams.",
    ],
    "review_metrics": [
        "Arm label rate (%) = macro_by_arm label rate × 100; count and denominator come from counts_by_arm.",
        "Per-facet rows preserve each per_facet_by_arm denominator, mutually exclusive relevance labels, and overlapping low_quality flag.",
    ],
    "raw_union": [
        "Each cutoff is the unique topic-qualified relevant union of original O@100 plus the named arm's facet streams at K.",
        "Headroom is the unique relevant raw-union document set absent from corrected C0's final top 100.",
    ],
    "facet_retention": [
        "Relevant at K is relevant_retained[K] for one frozen arm/facet stream; values are not added across overlapping facets.",
    ],
    "decision": [
        "Promotion-funnel counts are lengths of raw_union.headroom_docids, prefusion.pre_fusion_promoted_novel_docids, and prefusion.final_novel_docids, checked against decision.evidence.",
        "The promotion window is BF best facet rank <=20 and C0 best facet rank >20.",
        "Per-topic deltas come from decision.evidence.per_topic_deltas_vs_control; absolute C0 and BF100 nDCG@10 tooltips come from systems.",
        "Legacy attribution requires decision final_novel_vs_legacy_r1_docids to be contained in legacy_corrected_diff corrected_only sets.",
    ],
    "systems": [
        "Macro and per-topic report rows flatten each systems entry's metrics and per_topic objects without recomputation.",
    ],
    "gains_losses": [
        "Gained and lost are set cardinalities; net change must equal gained minus lost for BF100 versus each baseline.",
    ],
    "representatives": [
        "At most three saved rows per promoted/demoted/gained/lost class; passage text is bounded to 700 characters.",
    ],
    "representative_provenance": [
        "Each row is an offline derivation from the saved representative class, qualifying pre-fusion facet event, and authenticated frozen MiniLM stream.",
        "Facet C0/BF ranks and the selected highest-logit MiniLM window are preserved; no qrels projection was reopened.",
    ],
    "preflight": [
        "Coverage percentages equal saved coverage fractions × 100; model identity and window cap are copied from authenticated preflight fields.",
    ],
    "scoring_receipt": [
        "Planned, completed, unique-score, and failed-window counts are copied from the completed scoring receipt.",
    ],
    "benchmark": [
        "Pairs per second and projected wall time are copied from benchmark telemetry; device bytes are divided by 1,000,000 for MB.",
    ],
    "qrels_access_receipt": [
        "Qrels access count is one iff the bound receipt status is qrels_access_consumed.",
        "Approval-scoped consumption is true only when the stable registry binds the approval, receipt, freezes, projection hashes, and canonical evaluation output.",
        "Ranking/review frozen flags require qrels_opened=false in their authenticated freezes; shared memberships = membership_count - item_count.",
    ],
    "qrels_consumption_legacy_marker": [
        "This approval-adjacent v1 marker is retained for history only and is not trusted as the replay guard.",
    ],
}


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_representative_provenance(
    representatives: Mapping[str, object],
) -> None:
    for evidence_class in ("promoted", "demoted", "gained", "lost"):
        rows = _list(
            representatives.get(evidence_class),
            f"representative provenance {evidence_class}",
        )
        for raw_row in rows:
            row = _object(raw_row, "representative provenance row")
            topic_id = str(row.get("topic_id", ""))
            if topic_id not in PILOT_TOPIC_SET:
                raise ValueError("representative provenance topic boundary differs")
            if row.get("evidence_class") != evidence_class:
                raise ValueError("representative evidence class binding differs")
            window = _object(
                row.get("selected_minilm_window"),
                "representative selected MiniLM window",
            )
            window_text = window.get("window_text")
            if not isinstance(window_text, str) or not window_text:
                raise ValueError("representative selected window text is missing")
            if row.get("passage") != window_text:
                raise ValueError(
                    "representative passage differs from selected window text"
                )
            stream = _object(
                row.get("stream_provenance"),
                "representative stream provenance",
            )
            if not stream:
                raise ValueError("representative stream provenance is empty")
            if evidence_class == "promoted":
                bf_rank = _integer(
                    row.get("facet_bf_rank"), "representative BF facet rank"
                )
                control_rank = _integer(
                    row.get("facet_control_rank"),
                    "representative C0 facet rank",
                )
                if not (bf_rank <= 20 < control_rank):
                    raise ValueError("representative promoted predicate differs")
            if evidence_class == "demoted":
                control_rank = _integer(
                    row.get("c0_final_rank"), "representative C0 final rank"
                )
                bf_rank = _integer(
                    row.get("bf_final_rank"), "representative BF final rank"
                )
                if not control_rank < bf_rank:
                    raise ValueError("representative demoted predicate differs")


def _compact_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _set_hash(values: Sequence[object]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    ).hexdigest()


def _validate_loaded_sources(
    artifacts: Mapping[str, object], artifact_bytes: Mapping[str, bytes]
) -> dict[str, Mapping[str, object]]:
    if set(artifacts) != set(SOURCE_PATHS) or set(artifact_bytes) != set(SOURCE_PATHS):
        raise ValueError("saved report source boundary differs")
    loaded: dict[str, Mapping[str, object]] = {}
    for key in SOURCE_PATHS:
        payload = _object(artifacts[key], key)
        raw = artifact_bytes[key]
        if not isinstance(raw, bytes):
            raise ValueError(f"{key} loaded bytes are required")
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{key} loaded bytes are invalid JSON") from exc
        if decoded != payload:
            raise ValueError(f"{key} payload differs from loaded bytes")
        loaded[key] = payload
    return loaded


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    claimed = _sha256(payload.get(field), f"{label} self-hash")
    without_hash = dict(payload)
    without_hash.pop(field, None)
    if field == "review_freeze_sha256":
        encoded = _pretty_bytes(without_hash)
    elif field == "freeze_sha256":
        encoded = _compact_bytes(without_hash, newline=True)
    else:
        encoded = _compact_bytes(without_hash)
    actual = hashlib.sha256(encoded).hexdigest()
    if claimed != actual:
        raise ValueError(f"{label} self-hash differs")
    return claimed


def _validate_contract(
    loaded: Mapping[str, Mapping[str, object]],
    raw: Mapping[str, bytes],
    decision: Mapping[str, object],
) -> None:
    if decision != loaded["decision"]:
        raise ValueError("decision argument differs from authenticated decision")

    manifest = loaded["manifest"]
    streams = [_object(row, "manifest stream") for row in _list(manifest.get("streams"), "manifest streams")]
    if (
        manifest.get("schema_version") != "facet-local-minilm-manifest-v1"
        or manifest.get("status") != "frozen_source_snapshot"
        or tuple(manifest.get("topic_ids", ())) != PILOT_TOPICS
        or manifest.get("stream_count") != 31
        or manifest.get("candidate_rows") != 3100
        or len(streams) != 31
        or sum(row.get("family") == "original" for row in streams) != 4
        or sum(row.get("family") == "facet" for row in streams) != 27
        or any(str(row.get("topic_id")) not in PILOT_TOPIC_SET for row in streams)
    ):
        raise ValueError("manifest pilot boundary differs")
    if any(
        row.get("expected_rows") != 100
        or not isinstance(row.get("query"), str)
        or not row.get("query")
        for row in streams
    ):
        raise ValueError("manifest stream accounting differs")

    preflight = loaded["preflight"]
    scoring = loaded["scoring_receipt"]
    benchmark = loaded["benchmark"]
    freeze = loaded["ranking_freeze"]
    review = loaded["review_freeze"]
    create_receipt = loaded["review_create_receipt"]
    preflight_sha = hashlib.sha256(raw["preflight"]).hexdigest()
    scoring_sha = hashlib.sha256(raw["scoring_receipt"]).hexdigest()
    if (
        preflight.get("schema_version") != "facet-local-minilm-preflight-v2"
        or preflight.get("status") != "tokenizer_only_preflight_complete"
        or scoring.get("status") != "complete"
        or benchmark.get("status") != "benchmark_complete"
    ):
        raise ValueError("preflight, benchmark, or scoring status differs")
    if scoring.get("preflight_sha256") != preflight_sha:
        raise ValueError("scoring receipt preflight binding differs")
    if benchmark.get("preflight_sha256") != preflight_sha:
        raise ValueError("benchmark preflight binding differs")
    if scoring.get("model_materialization_receipt_sha256") != preflight.get(
        "model_materialization_receipt_sha256"
    ):
        raise ValueError("model materialization binding differs")

    model = _object(preflight.get("model_materialization"), "model materialization")
    safe_files = _list(model.get("allow_patterns"), "model safe-file allowlist")
    if (
        model.get("model_id") != "cross-encoder/ms-marco-MiniLM-L6-v2"
        or model.get("revision") != "c5ee24cb16019beea0893ab7796b1df96625c6b8"
        or model.get("status") != "pinned_safe_files_materialized"
        or len(safe_files) != 6
        or scoring.get("model") != model.get("model_id")
        or scoring.get("model_revision") != model.get("revision")
    ):
        raise ValueError("pinned MiniLM model identity differs")

    approvals = {
        "model_download_approval": model.get("approval_sha256"),
        "benchmark_approval": benchmark.get("benchmark_approval_sha256"),
        "full_scoring_approval": scoring.get("full_inference_approval_sha256"),
    }
    for key, claimed in approvals.items():
        if claimed != hashlib.sha256(raw[key]).hexdigest():
            raise ValueError(f"{key} binding differs")
    full_approval = loaded["full_scoring_approval"]
    if full_approval.get("benchmark_telemetry_sha256") != hashlib.sha256(
        raw["benchmark"]
    ).hexdigest():
        raise ValueError("full scoring approval benchmark binding differs")

    ranking_hash = _verify_self_hash(freeze, "freeze_sha256", "ranking freeze")
    ranking_bindings = _object(freeze.get("bindings"), "ranking freeze bindings")
    ranking_artifacts = _object(freeze.get("artifacts"), "ranking freeze artifacts")
    diff_artifact = _object(
        ranking_artifacts.get("legacy_corrected_diff.json"),
        "legacy corrected diff ranking artifact",
    )
    control_diff = loaded["legacy_corrected_diff"]
    if (
        freeze.get("status") != "frozen_before_qrels"
        or freeze.get("qrels_opened") is not False
        or tuple(freeze.get("topic_ids", ())) != PILOT_TOPICS
        or ranking_bindings.get("manifest_sha256")
        != hashlib.sha256(raw["manifest"]).hexdigest()
        or ranking_bindings.get("preflight_sha256") != preflight_sha
        or ranking_bindings.get("scoring_receipt_sha256") != scoring_sha
        or ranking_bindings.get("retrieval_call_count") != 0
        or ranking_bindings.get("inference_call_count") != 0
        or ranking_bindings.get("qrels_opened") is not False
        or control_diff.get("schema_version")
        != "facet-local-minilm-control-diff-v1"
        or control_diff.get("qrels_opened") is not False
        or diff_artifact.get("sha256")
        != hashlib.sha256(raw["legacy_corrected_diff"]).hexdigest()
        or diff_artifact.get("bytes") != len(raw["legacy_corrected_diff"])
    ):
        if (
            control_diff.get("schema_version")
            != "facet-local-minilm-control-diff-v1"
            or control_diff.get("qrels_opened") is not False
            or diff_artifact.get("sha256")
            != hashlib.sha256(raw["legacy_corrected_diff"]).hexdigest()
            or diff_artifact.get("bytes") != len(raw["legacy_corrected_diff"])
        ):
            raise ValueError("legacy corrected diff authentication differs")
        raise ValueError("ranking freeze authentication differs")

    review_hash = _verify_self_hash(
        review, "review_freeze_sha256", "review freeze"
    )
    review_bindings = _object(review.get("bindings"), "review freeze bindings")
    if (
        review.get("status") != "review_frozen_before_qrels"
        or review.get("qrels_opened") is not False
        or review_bindings.get("ranking_freeze_sha256") != ranking_hash
        or review_bindings.get("create_receipt_sha256")
        != hashlib.sha256(raw["review_create_receipt"]).hexdigest()
        or create_receipt.get("ranking_freeze_sha256") != ranking_hash
        or create_receipt.get("qrels_opened") is not False
        or create_receipt.get("membership_count") != 108
        or create_receipt.get("item_count") != 101
        or create_receipt.get("facet_count") != 27
    ):
        raise ValueError("blinded review freeze authentication differs")

    bindings: Mapping[str, object] | None = None
    for key in EVALUATION_KEYS:
        payload = loaded[key]
        _verify_self_hash(payload, "artifact_sha256", key)
        current = _object(payload.get("bindings"), f"{key} bindings")
        if bindings is None:
            bindings = current
        elif current != bindings:
            raise ValueError("evaluation artifact bindings differ")
    assert bindings is not None
    receipt = loaded["qrels_access_receipt"]
    _verify_self_hash(receipt, "artifact_sha256", "qrels access receipt")
    approval = loaded["qrels_access_approval"]
    registry = loaded["qrels_consumption_registry"]
    legacy_marker = loaded["qrels_consumption_legacy_marker"]
    _verify_self_hash(
        registry, "artifact_sha256", "qrels consumption registry"
    )
    _verify_self_hash(
        legacy_marker, "artifact_sha256", "legacy qrels consumption marker"
    )
    registry_identity = {
        "experiment_id": "rag25_facet_local_minilm_v1",
        "qrels_approval_sha256": hashlib.sha256(
            raw["qrels_access_approval"]
        ).hexdigest(),
        **{
            field: registry.get(field)
            for field in (
                "qrels_manifest_sha256",
                "qrels_projection_sha256",
                "ranking_freeze_sha256",
                "review_freeze_sha256",
            )
        },
    }
    canonical_evaluation = (
        REPO_ROOT / "outputs/rag25_facet_local_minilm_v1/evaluation_v1"
    ).resolve()
    if (
        bindings.get("ranking_freeze_sha256") != ranking_hash
        or bindings.get("review_freeze_sha256") != review_hash
        or receipt.get("ranking_freeze_sha256") != ranking_hash
        or receipt.get("review_freeze_sha256") != review_hash
        or receipt.get("status") != "qrels_access_consumed"
        or tuple(receipt.get("topic_ids", ())) != PILOT_TOPICS
        or approval.get("schema_version") != "pilot-qrels-access-approval-v1"
        or approval.get("status") != "approved"
        or tuple(approval.get("topic_ids", ())) != PILOT_TOPICS
        or registry.get("schema_version")
        != "facet-local-minilm-qrels-consumption-v2"
        or registry.get("status") != "qrels_access_consumed"
        or registry.get("registry_namespace")
        != "git_common_dir_path_independent_identity"
        or registry.get("experiment_id") != "rag25_facet_local_minilm_v1"
        or registry.get("identity_sha256")
        != hashlib.sha256(_compact_bytes(registry_identity)).hexdigest()
        or tuple(registry.get("topic_ids", ())) != PILOT_TOPICS
        or registry.get("qrels_approval_sha256")
        != hashlib.sha256(raw["qrels_access_approval"]).hexdigest()
        or registry.get("canonical_output_path") != str(canonical_evaluation)
        or registry.get("output_receipt_path")
        != str((canonical_evaluation / "qrels_access_receipt.json").resolve())
        or legacy_marker.get("schema_version")
        != "facet-local-minilm-qrels-consumption-v1"
        or legacy_marker.get("status") != "qrels_access_consumed"
        or any(
            receipt.get(field) != bindings.get(field)
            for field in (
                "qrels_manifest_sha256",
                "qrels_projection_sha256",
                "ranking_freeze_sha256",
                "review_freeze_sha256",
            )
        )
        or any(
            registry.get(field) != receipt.get(field)
            or approval.get(field) != receipt.get(field)
            for field in (
                "qrels_manifest_sha256",
                "qrels_projection_sha256",
                "ranking_freeze_sha256",
                "review_freeze_sha256",
            )
        )
    ):
        raise ValueError("one-time qrels evaluation binding differs")

    provenance = loaded["representative_provenance"]
    provenance_hash = _verify_self_hash(
        provenance, "artifact_sha256", "representative provenance"
    )
    provenance_bindings = _object(
        provenance.get("bindings"), "representative provenance bindings"
    )
    if (
        not provenance_hash
        or provenance.get("schema_version")
        != "facet-local-minilm-representative-provenance-v2"
        or provenance.get("status") != "offline_derived_from_frozen_artifacts"
        or tuple(provenance.get("topic_ids", ())) != PILOT_TOPICS
        or provenance_bindings.get("ranking_freeze_sha256") != ranking_hash
        or provenance_bindings.get("review_freeze_sha256") != review_hash
        or provenance_bindings.get("source_prefusion_artifact_sha256")
        != loaded["prefusion"].get("artifact_sha256")
        or provenance_bindings.get("source_representatives_artifact_sha256")
        != loaded["representatives"].get("artifact_sha256")
    ):
        raise ValueError("representative provenance authentication differs")
    _validate_representative_provenance(provenance)
    for evidence_class in ("promoted", "demoted", "gained", "lost"):
        source_rows = _list(
            loaded["representatives"].get(evidence_class),
            f"source representatives {evidence_class}",
        )
        derived_rows = _list(
            provenance.get(evidence_class),
            f"derived representatives {evidence_class}",
        )
        source_ids = [
            (str(_object(row, "source representative").get("topic_id")),
             str(_object(row, "source representative").get("document_id")))
            for row in source_rows
        ]
        derived_ids = [
            (str(_object(row, "derived representative").get("topic_id")),
             str(_object(row, "derived representative").get("document_id")))
            for row in derived_rows
        ]
        if source_ids != derived_ids:
            raise ValueError("representative provenance class membership differs")


def _source_sql(source_id: str, paths: Sequence[str]) -> str:
    path = paths[0]
    if source_id == "manifest":
        return f"""WITH source AS (
  SELECT content::JSON AS doc FROM read_text('{path}')
)
SELECT
  json_extract_string(stream.value, '$.topic_id') AS topic_id,
  json_extract_string(stream.value, '$.family') AS family,
  json_extract_string(stream.value, '$.variant') AS variant,
  json_extract_string(stream.value, '$.query') AS full_query,
  CAST(json_extract(stream.value, '$.expected_rows') AS INTEGER) AS candidate_rows,
  json_extract_string(stream.value, '$.query_sha256') AS query_sha256
FROM source, json_each(source.doc, '$.streams') AS stream;"""
    if source_id == "review_metrics":
        return f"""WITH source AS (
  SELECT content::JSON AS doc FROM read_text('{path}')
), arm_rows AS (
  SELECT arm.key AS arm_id, arm.value AS metrics
  FROM source, json_each(source.doc, '$.macro_by_arm') AS arm
), facet_arms AS (
  SELECT arm.key AS arm_id, arm.value AS facets
  FROM source, json_each(source.doc, '$.per_facet_by_arm') AS arm
), facet_rows AS (
  SELECT arm_id, facet.key AS facet_id, facet.value AS metrics
  FROM facet_arms, json_each(facet_arms.facets) AS facet
)
SELECT 'arm' AS grain, arm_id, NULL::VARCHAR AS facet_id, metrics FROM arm_rows
UNION ALL
SELECT 'facet' AS grain, arm_id, facet_id, metrics FROM facet_rows;"""
    if source_id == "raw_union":
        return f"""WITH source AS (
  SELECT content::JSON AS doc FROM read_text('{path}')
), arms AS (
  SELECT arm.key AS arm_id, arm.value AS depths
  FROM source, json_each(source.doc, '$.curves') AS arm
)
SELECT arm_id, CAST(depth.key AS INTEGER) AS facet_depth,
       json_extract(depth.value, '$.aggregate') AS aggregate_metrics
FROM arms, json_each(arms.depths) AS depth
WHERE arm_id IN ('C0_TOPIC_LOCAL', 'BF100_TOPIC_LOCAL')
  AND CAST(depth.key AS INTEGER) IN (20, 50, 100);"""
    if source_id == "facet_retention":
        return f"""WITH source AS (
  SELECT content::JSON AS doc FROM read_text('{path}')
), arms AS (
  SELECT arm.key AS arm_id, arm.value AS facets
  FROM source, json_each(source.doc, '$.arms') AS arm
)
SELECT arm_id, facet.key AS facet_id,
       CAST(json_extract(facet.value, '$.relevant_retained.20') AS INTEGER) AS relevant_at_k20,
       CAST(json_extract(facet.value, '$.relevant_retained.50') AS INTEGER) AS relevant_at_k50,
       CAST(json_extract(facet.value, '$.relevant_retained.100') AS INTEGER) AS relevant_at_k100
FROM arms, json_each(arms.facets) AS facet;"""
    if source_id == "decision":
        decision_path, raw_path, prefusion_path, systems_path, diff_path, review_path = paths
        return f"""WITH joined AS (
  SELECT d.content::JSON AS decision, r.content::JSON AS raw_union,
         p.content::JSON AS prefusion, s.content::JSON AS systems,
         c.content::JSON AS control_diff, v.content::JSON AS review
  FROM read_text('{decision_path}') d
  CROSS JOIN read_text('{raw_path}') r
  CROSS JOIN read_text('{prefusion_path}') p
  CROSS JOIN read_text('{systems_path}') s
  CROSS JOIN read_text('{diff_path}') c
  CROSS JOIN read_text('{review_path}') v
)
SELECT
  json_array_length(json_extract(raw_union, '$.headroom_docids')) AS raw_headroom,
  json_array_length(json_extract(prefusion, '$.pre_fusion_promoted_novel_docids')) AS promoted_top20,
  json_array_length(json_extract(prefusion, '$.fusion_blocked_novel_docids')) AS fusion_blocked,
  json_array_length(json_extract(prefusion, '$.final_novel_docids')) AS final_novel,
  json_extract(decision, '$.evidence.per_topic_deltas_vs_control') AS per_topic_deltas,
  json_extract(systems, '$.systems.C0_TOPIC_LOCAL') AS c0_metrics,
  json_extract(systems, '$.systems.BF100_TOPIC_LOCAL') AS bf100_metrics,
  json_extract(decision, '$.evidence.final_novel_vs_legacy_r1_docids') AS novel_vs_legacy,
  json_extract(control_diff, '$.topics') AS corrected_legacy_audit,
  json_extract(review, '$.macro_by_arm') AS blinded_review_rates
FROM joined;"""
    if source_id == "systems":
        return f"""WITH source AS (
  SELECT content::JSON AS doc FROM read_text('{path}')
)
SELECT system.key AS system_id,
       json_extract(system.value, '$.metrics') AS macro_metrics,
       json_extract(system.value, '$.per_topic') AS per_topic_metrics
FROM source, json_each(source.doc, '$.systems') AS system;"""
    if source_id == "gains_losses":
        return f"""WITH source AS (
  SELECT content::JSON AS doc FROM read_text('{path}')
), candidates AS (
  SELECT candidate.key AS candidate_id, candidate.value AS comparisons
  FROM source, json_each(source.doc, '$.systems') AS candidate
)
SELECT candidate_id, baseline.key AS baseline_id,
       json_extract(baseline.value, '$.aggregate') AS aggregate,
       json_extract(baseline.value, '$.per_topic') AS per_topic
FROM candidates, json_each(candidates.comparisons) AS baseline
WHERE candidate_id = 'BF100_TOPIC_LOCAL'
  AND baseline.key IN ('C0_TOPIC_LOCAL', 'R1_LEGACY');"""
    if source_id in {"representatives", "representative_provenance"}:
        return f"""WITH source AS (
  SELECT content::JSON AS doc FROM read_text('{path}')
), classes(evidence_class, rows) AS (
  SELECT 'promoted', json_extract(doc, '$.promoted') FROM source UNION ALL
  SELECT 'demoted', json_extract(doc, '$.demoted') FROM source UNION ALL
  SELECT 'gained', json_extract(doc, '$.gained') FROM source UNION ALL
  SELECT 'lost', json_extract(doc, '$.lost') FROM source
)
SELECT evidence_class, item.key AS saved_order, item.value AS saved_evidence
FROM classes, json_each(classes.rows) AS item;"""
    if source_id == "preflight":
        return f"""SELECT model_materialization.model_id AS model,
       model_materialization.revision AS revision,
       model_materialization.allow_patterns AS safe_files,
       window_policy.maximum_windows_per_document AS window_cap,
       summary.capped_document_count AS capped_documents,
       summary.coverage_min * 100 AS coverage_min_percent,
       summary.coverage_p95 * 100 AS coverage_p95_percent
FROM read_json_auto('{path}');"""
    if source_id == "scoring_receipt":
        return f"""SELECT model, model_revision AS revision, inference_dtype AS dtype,
       planned_window_count AS planned_windows,
       completed_window_count AS completed_windows,
       unique_score_count AS unique_scores, failed_window_count AS failed_windows
FROM read_json_auto('{path}');"""
    if source_id == "benchmark":
        return f"""SELECT execution_backend AS backend, device_probe.device_name AS device,
       median_pairs_per_second AS pairs_per_second,
       peak_device_memory_bytes / 1000000.0 AS peak_device_memory_mb,
       projected_full_run_wall_seconds AS projected_wall_seconds
FROM read_json_auto('{path}');"""
    if source_id == "qrels_access_receipt":
        (
            receipt_path,
            approval_path,
            registry_path,
            ranking_path,
            review_path,
            create_path,
        ) = paths
        return f"""SELECT q.status AS qrels_access_status, 1 AS qrels_access_count,
       g.status = 'qrels_access_consumed'
         AND g.qrels_approval_sha256 = sha256(read_blob('{approval_path}'))
         AS approval_scoped_consumption_registry,
       g.registry_namespace AS registry_namespace,
       false AS legacy_marker_trusted,
       r.qrels_opened = false AS ranking_frozen_before_qrels,
       v.qrels_opened = false AS review_frozen_before_qrels,
       c.item_count AS review_unique_items, c.membership_count AS review_memberships,
       c.membership_count - c.item_count AS shared_items_attributed_to_both_arms
FROM read_json_auto('{receipt_path}') q
CROSS JOIN read_json_auto('{registry_path}') g
CROSS JOIN read_json_auto('{ranking_path}') r
CROSS JOIN read_json_auto('{review_path}') v
CROSS JOIN read_json_auto('{create_path}') c
WHERE q.status = 'qrels_access_consumed'
  AND q.ranking_freeze_sha256 = r.freeze_sha256
  AND q.review_freeze_sha256 = v.review_freeze_sha256;"""
    return f"SELECT * FROM read_json_auto('{path}');"


def _source(source_id: str, raw_sources: Mapping[str, bytes]) -> dict[str, object]:
    path = SOURCE_PATHS[source_id]
    material_ids = MATERIAL_SOURCE_IDS.get(source_id, (source_id,))
    paths = [SOURCE_PATHS[key] for key in material_ids]
    input_sha256 = {
        SOURCE_PATHS[key]: hashlib.sha256(raw_sources[key]).hexdigest()
        for key in material_ids
    }
    if len(input_sha256) == 1:
        digest = next(iter(input_sha256.values()))
    else:
        digest = hashlib.sha256(_compact_bytes(input_sha256)).hexdigest()
    return {
        "id": source_id,
        "label": SOURCE_LABELS[source_id],
        "path": path,
        "query": {
            "engine": "duckdb",
            "id": f"sha256:{digest}",
            "language": "sql",
            "description": f"Extracts the report rows from the authenticated {SOURCE_LABELS[source_id].lower()} input boundary.",
            "sql": _source_sql(source_id, paths),
            "tables_used": paths,
            "input_sha256": input_sha256,
            "filters": [
                "Only topics 200, 225, 707, and 897",
                "Protected topics, raw qrels rows, and private review mappings excluded",
                "Saved artifact bytes authenticated before report construction",
            ],
            "metric_definitions": SOURCE_METRIC_DEFINITIONS.get(
                source_id,
                ["Saved artifact fields are loaded without analytical recomputation."],
            ),
        },
    }


def _table(
    table_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    source_id: str,
    columns: Sequence[tuple[str, str, str]],
    sort_field: str,
    *,
    density: str = "spacious",
) -> dict[str, object]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "sourceId": source_id,
        "density": density,
        "layout": "full",
        "defaultSort": {"field": sort_field, "direction": "asc"},
        "columns": [
            {"field": field, "label": label, "type": kind}
            for field, label, kind in columns
        ],
    }


def _review_rows(review_metrics: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    macro = _object(review_metrics.get("macro_by_arm"), "review macro metrics")
    counts = _object(review_metrics.get("counts_by_arm"), "review counts")
    labels = (
        ("Direct answer", "direct_answer", "direct_answer_rate"),
        ("Partial or related", "partial_or_related", "partial_or_related_rate"),
        ("Irrelevant to facet", "not_facet_relevant", "not_facet_relevant_rate"),
        ("Wrong domain", "wrong_domain", "wrong_domain_rate"),
        ("Low quality", "low_quality", "low_quality_rate"),
    )
    arm_names = {
        "C0_TOPIC_LOCAL": "C0",
        "BF50_TOPIC_LOCAL": "BF50",
    }
    if set(macro) != set(arm_names) or set(counts) != set(arm_names):
        raise ValueError("blinded review arm boundary differs")
    aggregate_rows: list[dict[str, object]] = []
    for arm_id, arm in arm_names.items():
        metric_row = _object(macro[arm_id], f"review metrics {arm_id}")
        count_row = _object(counts[arm_id], f"review counts {arm_id}")
        denominator = _integer(count_row.get("denominator"), f"{arm_id} denominator")
        if denominator != 54:
            raise ValueError("blinded review denominator differs")
        for label, count_field, rate_field in labels:
            count = _integer(count_row.get(count_field), f"{arm_id} {count_field}")
            rate = _number(metric_row.get(rate_field), f"{arm_id} {rate_field}")
            if not math.isclose(rate, count / denominator, abs_tol=1e-15):
                raise ValueError(f"blinded review rate differs for {arm_id}/{label}")
            aggregate_rows.append(
                {
                    "arm": arm,
                    "arm_id": arm_id,
                    "label": label,
                    "count": count,
                    "denominator": denominator,
                    "rate_percent": rate * 100.0,
                }
            )

    per_facet = _object(review_metrics.get("per_facet_by_arm"), "per-facet review")
    if set(per_facet) != set(arm_names):
        raise ValueError("per-facet review arm boundary differs")
    facet_rows: list[dict[str, object]] = []
    expected_facets: set[str] | None = None
    for arm_id, arm in arm_names.items():
        by_facet = _object(per_facet[arm_id], f"per-facet review {arm_id}")
        if expected_facets is None:
            expected_facets = set(by_facet)
        elif set(by_facet) != expected_facets:
            raise ValueError("per-facet review identities differ by arm")
        for facet_key in sorted(by_facet):
            values = _object(by_facet[facet_key], f"per-facet review {facet_key}")
            topic_id, variant = facet_key.split("/", 1)
            if topic_id not in PILOT_TOPIC_SET or values.get("denominator") != 2:
                raise ValueError("per-facet review boundary differs")
            facet_rows.append(
                {
                    "topic_id": topic_id,
                    "facet": variant,
                    "arm": arm,
                    "denominator": values["denominator"],
                    "direct_answer": values["direct_answer"],
                    "direct_answer_percent": _number(values["direct_answer_rate"], "direct answer rate") * 100.0,
                    "partial_or_related": values["partial_or_related"],
                    "irrelevant": values["not_facet_relevant"],
                    "wrong_domain": values["wrong_domain"],
                    "low_quality": values["low_quality"],
                }
            )
    if expected_facets is None or len(expected_facets) != 27 or len(facet_rows) != 54:
        raise ValueError("per-facet review requires exact 27-by-two accounting")
    return aggregate_rows, facet_rows


def _system_rows(systems_artifact: Mapping[str, object]) -> list[dict[str, object]]:
    systems = _object(systems_artifact.get("systems"), "evaluated systems")
    expected = {
        "O",
        "R1_LEGACY",
        "C0_TOPIC_LOCAL",
        "BF100_TOPIC_LOCAL",
        "BF50_TOPIC_LOCAL",
        "BF20_TOPIC_LOCAL",
        "BO100_TOPIC_LOCAL",
        "BB100_TOPIC_LOCAL",
        "BF100_MAXP_TOPIC_LOCAL",
        "BF100_LEGACY_FUSION",
    }
    # O plus the nine preregistered ranking arms contains ten systems.
    if set(systems) != expected:
        raise ValueError("evaluated system boundary differs")
    rows: list[dict[str, object]] = []
    fields = (
        "ndcg@10",
        "graded_recall@100",
        "recall@100",
        "precision@10",
        "relevant_count@10",
        "relevant_count@100",
        "judged_rate@10",
        "judged_rate@100",
        "oracle_ndcg@10_from_top50",
        "oracle_ndcg@10_from_top100",
    )
    for system_id in sorted(systems):
        system = _object(systems[system_id], f"system {system_id}")
        macro = _object(system.get("metrics"), f"system metrics {system_id}")
        per_topic = _object(system.get("per_topic"), f"per-topic system {system_id}")
        if set(per_topic) != PILOT_TOPIC_SET:
            raise ValueError(f"system topic boundary differs for {system_id}")
        for scope, topic_id, values in (
            ("Macro", "All four", macro),
            *(("Topic", topic, _object(per_topic[topic], f"{system_id}/{topic}")) for topic in PILOT_TOPICS),
        ):
            row: dict[str, object] = {
                "row_label": f"{system_id} · {topic_id}",
                "scope": scope,
                "topic_id": topic_id,
                "system": system_id,
            }
            for field in fields:
                row[field.replace("@", "_at_").replace(".", "_")] = _number(
                    values.get(field), f"{system_id}/{topic_id}/{field}"
                )
            rows.append(row)
    return rows


def _representative_rows(representatives: Mapping[str, object]) -> list[dict[str, object]]:
    labels = {
        "promoted": "Promoted before fusion",
        "demoted": "Demoted in final ranking",
        "gained": "Gained in final ranking",
        "lost": "Lost from final ranking",
    }
    rows: list[dict[str, object]] = []
    for key, label in labels.items():
        values = [_object(value, f"representative {key}") for value in _list(representatives.get(key), f"representatives {key}")]
        if not values:
            rows.append(
                {
                    "evidence_class": label,
                    "topic_id": "All four",
                    "document_id": "None",
                    "facet": "n/a",
                    "c0_facet_rank": None,
                    "bf_facet_rank": None,
                    "passage": f"No {key} final-ranking passage exists in this comparison.",
                    "provenance": "The authenticated offline derivation contains an empty representative set.",
                    "source_artifact": "representative_provenance_v2.json",
                }
            )
            continue
        for value in values[:3]:
            topic_id = str(value.get("topic_id"))
            if topic_id not in PILOT_TOPIC_SET:
                raise ValueError("representative topic boundary differs")
            facet = value.get("facet_variant")
            control_rank = value.get("facet_control_rank")
            bf_rank = value.get("facet_bf_rank")
            if control_rank is not None:
                control_rank = _integer(control_rank, "representative C0 facet rank")
            if bf_rank is not None:
                bf_rank = _integer(bf_rank, "representative BF facet rank")
            stream = _object(
                value.get("stream_provenance", {}),
                "representative stream provenance",
            )
            window = value.get("selected_minilm_window")
            if window is not None:
                window = _object(window, "representative selected MiniLM window")
            provenance_text = (
                f"Facet {facet or 'n/a'} · C0 facet rank {control_rank or 'not retrieved'}"
                f" · BF facet rank {bf_rank or 'not retrieved'}"
                f" · MiniLM window {window.get('window_id', 'n/a') if window else 'n/a'}"
                f" · source BM25 rank {stream.get('prior_rank', 'n/a')}"
                f" · query SHA-256 {stream.get('query_sha256', 'n/a')}"
            )
            passage = str(value.get("passage", ""))
            rows.append(
                {
                    "evidence_class": label,
                    "topic_id": topic_id,
                    "document_id": str(value.get("document_id")),
                    "facet": str(facet or "n/a"),
                    "c0_facet_rank": control_rank,
                    "bf_facet_rank": bf_rank,
                    "passage": passage[:697] + ("…" if len(passage) > 697 else ""),
                    "provenance": provenance_text,
                    "source_artifact": "representative_provenance_v2.json",
                }
            )
    return rows


def _artifact_rows(loaded: Mapping[str, Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    manifest = loaded["manifest"]
    streams = [_object(row, "manifest stream") for row in _list(manifest.get("streams"), "manifest streams")]
    query_rows = [
        {
            "topic_id": str(stream["topic_id"]),
            "family": str(stream["family"]),
            "variant": str(stream["variant"]),
            "full_query": str(stream["query"]),
            "candidate_rows": int(stream["expected_rows"]),
            "query_sha256": str(stream["query_sha256"]),
        }
        for stream in streams
    ]
    original_by_topic = {
        row["topic_id"]: row for row in query_rows if row["family"] == "original"
    }
    facet_counts = {
        topic_id: sum(
            row["topic_id"] == topic_id and row["family"] == "facet"
            for row in query_rows
        )
        for topic_id in PILOT_TOPICS
    }
    narrative_rows = [
        {
            "topic_id": topic_id,
            "facet_count": facet_counts[topic_id],
            "full_narrative": original_by_topic[topic_id]["full_query"],
        }
        for topic_id in PILOT_TOPICS
    ]

    review_rate_rows, facet_review_rows = _review_rows(loaded["review_metrics"])
    raw_union = loaded["raw_union"]
    curves = _object(raw_union.get("curves"), "raw-union curves")
    candidate_union_rows: list[dict[str, object]] = []
    for arm_id, arm in (("C0_TOPIC_LOCAL", "C0"), ("BF100_TOPIC_LOCAL", "BF100")):
        depth_rows = _object(curves.get(arm_id), f"raw-union curve {arm_id}")
        for depth in (20, 50, 100):
            curve = _object(depth_rows.get(str(depth)), f"{arm_id}/K{depth}")
            aggregate = _object(curve.get("aggregate"), f"{arm_id}/K{depth} aggregate")
            candidate_union_rows.append(
                {
                    "arm": arm,
                    "arm_id": arm_id,
                    "depth": f"K{depth}",
                    "depth_value": depth,
                    "relevant_documents": _integer(aggregate.get("relevant_documents"), "relevant union count"),
                    "unique_candidate_documents": _integer(aggregate.get("unique_candidate_documents"), "unique union count"),
                    "macro_recall": _number(aggregate.get("macro_recall"), "union macro recall"),
                    "macro_graded_recall": _number(aggregate.get("macro_graded_recall"), "union graded recall"),
                }
            )

    retention_arms = _object(
        loaded["facet_retention"].get("arms"), "facet-retention arms"
    )
    if set(retention_arms) != {"C0_TOPIC_LOCAL", "BF100_TOPIC_LOCAL"}:
        raise ValueError("facet-retention arm boundary differs")
    expected_facets = {
        f"{row['topic_id']}/{row['variant']}"
        for row in query_rows
        if row["family"] == "facet"
    }
    facet_retention_rows: list[dict[str, object]] = []
    for arm_id, arm in (("C0_TOPIC_LOCAL", "C0"), ("BF100_TOPIC_LOCAL", "BF100")):
        arm_rows = _object(retention_arms.get(arm_id), f"retention arm {arm_id}")
        if set(arm_rows) != expected_facets:
            raise ValueError(f"facet-retention stream boundary differs for {arm_id}")
        for facet_key in sorted(arm_rows):
            topic_id, facet = facet_key.split("/", 1)
            retained = _object(
                _object(arm_rows[facet_key], f"retention {arm_id}/{facet_key}").get(
                    "relevant_retained"
                ),
                f"retained cutoffs {arm_id}/{facet_key}",
            )
            facet_retention_rows.append(
                {
                    "topic_id": topic_id,
                    "facet": facet,
                    "arm": arm,
                    "relevant_at_k20": _integer(retained.get("20"), "K20 retention"),
                    "relevant_at_k50": _integer(retained.get("50"), "K50 retention"),
                    "relevant_at_k100": _integer(retained.get("100"), "K100 retention"),
                }
            )

    decision = loaded["decision"]
    decision_values = _object(decision.get("decision"), "mechanical decision")
    evidence = _object(decision.get("evidence"), "decision evidence")
    prefusion = loaded["prefusion"]
    headroom_docids = _list(raw_union.get("headroom_docids"), "headroom document set")
    promoted_docids = _list(prefusion.get("pre_fusion_promoted_novel_docids"), "pre-fusion promoted set")
    final_novel_docids = _list(prefusion.get("final_novel_docids"), "final novel set")
    if (
        len(headroom_docids) != evidence.get("headroom")
        or len(promoted_docids) != evidence.get("pre_fusion_promoted_novel")
        or len(final_novel_docids) != evidence.get("final_novel")
        or len(_list(prefusion.get("fusion_blocked_novel_docids"), "fusion-blocked set")) != evidence.get("fusion_blocked_novel")
    ):
        raise ValueError("raw, pre-fusion, and final document-set accounting differs")
    stage_set_rows = [
        {"stage": "Raw-union headroom", "relevant_documents": len(headroom_docids), "document_set_sha256": _set_hash(headroom_docids)},
        {"stage": "Pre-fusion promoted novel", "relevant_documents": len(promoted_docids), "document_set_sha256": _set_hash(promoted_docids)},
        {"stage": "Final novel vs corrected C0", "relevant_documents": len(final_novel_docids), "document_set_sha256": _set_hash(final_novel_docids)},
    ]

    system_rows = _system_rows(loaded["systems"])
    system_index = {(row["system"], row["topic_id"]): row for row in system_rows}
    deltas = _object(evidence.get("per_topic_deltas_vs_control"), "per-topic deltas")
    topic_ndcg_rows = []
    for topic_id in PILOT_TOPICS:
        delta = _object(deltas.get(topic_id), f"topic delta {topic_id}")
        topic_ndcg_rows.append(
            {
                "topic_id": topic_id,
                "ndcg_delta": _number(delta.get("ndcg@10"), f"topic {topic_id} nDCG delta"),
                "recall_delta": _number(delta.get("recall@100"), f"topic {topic_id} recall delta"),
                "graded_recall_delta": _number(delta.get("graded_recall@100"), f"topic {topic_id} graded recall delta"),
                "c0_ndcg_at_10": system_index[("C0_TOPIC_LOCAL", topic_id)]["ndcg_at_10"],
                "bf100_ndcg_at_10": system_index[("BF100_TOPIC_LOCAL", topic_id)]["ndcg_at_10"],
            }
        )

    gain_loss = _object(_object(loaded["gains_losses"].get("systems"), "gain/loss systems").get("BF100_TOPIC_LOCAL"), "BF100 gain/loss comparisons")
    gain_loss_rows: list[dict[str, object]] = []
    for baseline in ("C0_TOPIC_LOCAL", "R1_LEGACY"):
        comparison = _object(gain_loss.get(baseline), f"gain/loss versus {baseline}")
        scoped = [("Aggregate", "All four", _object(comparison.get("aggregate"), "aggregate gain/loss"))]
        per_topic = _object(comparison.get("per_topic"), "per-topic gain/loss")
        scoped.extend(("Topic", topic_id, _object(per_topic.get(topic_id), f"gain/loss {topic_id}")) for topic_id in PILOT_TOPICS)
        for scope, topic_id, values in scoped:
            gained = len(_list(values.get("gained"), "gained documents"))
            lost = len(_list(values.get("lost"), "lost documents"))
            net = _integer(values.get("net_change"), "net relevant change")
            if net != gained - lost:
                raise ValueError("gain/loss net accounting differs")
            gain_loss_rows.append(
                {
                    "row_label": f"BF100 vs {baseline} · {topic_id}",
                    "scope": scope,
                    "topic_id": topic_id,
                    "candidate": "BF100_TOPIC_LOCAL",
                    "baseline": baseline,
                    "baseline_relevant": values["baseline_count"],
                    "candidate_relevant": values["candidate_count"],
                    "gained": gained,
                    "lost": lost,
                    "net_change": net,
                }
            )

    preflight = loaded["preflight"]
    summary = _object(preflight.get("summary"), "preflight summary")
    window_policy = _object(preflight.get("window_policy"), "window policy")
    model = _object(preflight.get("model_materialization"), "model materialization")
    benchmark = loaded["benchmark"]
    scoring = loaded["scoring_receipt"]
    device_probe = _object(benchmark.get("device_probe"), "benchmark device probe")
    runtime_rows = [
        {
            "model": model["model_id"],
            "revision": model["revision"],
            "safe_file_count": len(_list(model.get("allow_patterns"), "safe files")),
            "safe_files_only": True,
            "execution_backend": benchmark["execution_backend"],
            "device": device_probe["device_name"],
            "inference_dtype": scoring["inference_dtype"],
            "planned_windows": scoring["planned_window_count"],
            "completed_windows": scoring["completed_window_count"],
            "unique_scored_pairs": scoring["unique_score_count"],
            "capped_documents": summary["capped_document_count"],
            "maximum_windows_per_document": window_policy["maximum_windows_per_document"],
            "coverage_min_percent": _number(summary["coverage_min"], "minimum coverage") * 100.0,
            "coverage_median_percent": _number(summary["coverage_median"], "median coverage") * 100.0,
            "coverage_p95_percent": _number(summary["coverage_p95"], "p95 coverage") * 100.0,
        }
    ]
    model_rows = [
        {
            "model": model["model_id"],
            "revision": model["revision"],
            "safe_files": ", ".join(str(value) for value in model["allow_patterns"]),
            "window_cap": window_policy["maximum_windows_per_document"],
            "capped_documents": summary["capped_document_count"],
            "coverage_min_percent": runtime_rows[0]["coverage_min_percent"],
            "coverage_p95_percent": runtime_rows[0]["coverage_p95_percent"],
        }
    ]
    scoring_rows = [
        {
            "model": scoring["model"],
            "revision": scoring["model_revision"],
            "dtype": scoring["inference_dtype"],
            "planned_windows": scoring["planned_window_count"],
            "completed_windows": scoring["completed_window_count"],
            "unique_scores": scoring["unique_score_count"],
            "failed_windows": scoring["failed_window_count"],
        }
    ]
    benchmark_rows = [
        {
            "backend": benchmark["execution_backend"],
            "device": device_probe["device_name"],
            "pairs_per_second": benchmark["median_pairs_per_second"],
            "peak_device_memory_mb": benchmark["peak_device_memory_bytes"] / 1_000_000,
            "projected_wall_seconds": benchmark["projected_full_run_wall_seconds"],
        }
    ]

    receipt = loaded["qrels_access_receipt"]
    registry = loaded["qrels_consumption_registry"]
    create = loaded["review_create_receipt"]
    firewall_rows = [
        {
            "qrels_access_status": receipt["status"],
            "qrels_access_count": 1,
            "approval_scoped_consumption_registry": (
                registry["status"] == "qrels_access_consumed"
            ),
            "registry_namespace": registry["registry_namespace"],
            "legacy_marker_trusted": False,
            "ranking_frozen_before_qrels": loaded["ranking_freeze"]["qrels_opened"] is False,
            "review_frozen_before_qrels": loaded["review_freeze"]["qrels_opened"] is False,
            "review_unique_items": create["item_count"],
            "review_memberships": create["membership_count"],
            "shared_items_attributed_to_both_arms": create["membership_count"] - create["item_count"],
        }
    ]
    return {
        "narrative_rows": narrative_rows,
        "query_rows": query_rows,
        "review_rate_rows": review_rate_rows,
        "facet_review_rows": facet_review_rows,
        "candidate_union_rows": candidate_union_rows,
        "facet_retention_rows": facet_retention_rows,
        "stage_set_rows": stage_set_rows,
        "topic_ndcg_rows": topic_ndcg_rows,
        "system_rows": system_rows,
        "gain_loss_rows": gain_loss_rows,
        "representative_rows": _representative_rows(
            loaded["representative_provenance"]
        ),
        "runtime_rows": runtime_rows,
        "model_rows": model_rows,
        "scoring_rows": scoring_rows,
        "benchmark_rows": benchmark_rows,
        "firewall_rows": firewall_rows,
    }


def build_artifact(
    *,
    artifacts: Mapping[str, object],
    artifact_bytes: Mapping[str, bytes],
    decision: Mapping[str, object],
) -> dict[str, object]:
    """Return one deterministic, source-backed portable report artifact."""

    loaded = _validate_loaded_sources(artifacts, artifact_bytes)
    decision = _object(decision, "decision")
    _validate_contract(loaded, artifact_bytes, decision)
    rows = _artifact_rows(loaded)
    decision_values = _object(loaded["decision"].get("decision"), "decision")
    evidence = _object(loaded["decision"].get("evidence"), "decision evidence")
    outcome = str(decision_values["outcome"])
    macro = _object(evidence.get("macro_deltas_vs_control"), "macro deltas")
    legacy_novel = _list(
        evidence.get("final_novel_vs_legacy_r1_docids"),
        "final novel versus legacy set",
    )
    control_diff = _object(
        loaded["legacy_corrected_diff"].get("topics"), "fusion correction audit"
    )
    corrected_only = {
        f"{topic_id}/{docid}"
        for topic_id in PILOT_TOPICS
        for docid in _list(
            _object(control_diff.get(topic_id), f"fusion audit {topic_id}").get(
                "corrected_only"
            ),
            f"corrected-only documents {topic_id}",
        )
    }
    if len(legacy_novel) != 2 or not set(legacy_novel) <= corrected_only:
        raise ValueError("legacy-R1 novelty is not attributable to fusion correction")

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
            "sourceId": "decision",
            "body": (
                "## Technical summary\n\n"
                f"**MiniLM materially cleans facet-local results, but the current family-balanced reciprocal-rank fusion (RRF) erases the coverage gain.** The raw union contains **220 relevant documents absent from corrected C0's final top 100**; reranking promotes **22** of them before fusion, yet **0** remain novel after fusion. The mechanical outcome is `{outcome}`.\n\n"
                "**Fix fusion before Stage A or new retrieval.** Blinded review shows that BF50 improves direct answers and reduces wrong-domain, irrelevant, and low-quality results. Because useful candidates already exist and MiniLM promotes some locally, candidate absence and local reranking are not the binding failure; the final fusion block is. Stage A was not permitted and was not executed.\n\n"
                "**nDCG@10 is a guardrail, not evidence of coverage.** BF100 changes macro nDCG@10 by only **+0.0025617**, while macro Recall@100 and graded Recall@100 are unchanged. This is a four-topic descriptive pilot with no inferential claim."
            ),
        },
        {
            "id": "review_heading",
            "type": "markdown",
            "layout": "full",
            "sourceId": "review_metrics",
            "body": (
                "## Blinded review shows cleaner facet-local evidence\n\n"
                "**BF50's direct-answer rate is 75.93% versus 59.26% for C0 (+16.67 percentage points).** Wrong-domain falls to 1.85% from 7.41% (-5.56 points), irrelevant-to-facet to 5.56% from 18.52%, and low-quality to 31.48% from 38.89%. Rates use 54 arm/facet memberships per arm; quality flags can overlap relevance labels."
            ),
        },
        {
            "id": "review_chart_block",
            "type": "chart",
            "chartId": "facet_review_rates",
            "layout": "full",
        },
        {
            "id": "candidate_heading",
            "type": "markdown",
            "layout": "full",
            "sourceId": "raw_union",
            "body": (
                "## Candidate coverage improves early, then converges by K100\n\n"
                "**The BF100 union retains more relevant documents at the shallower cutoffs:** 295 versus 292 at K20 and 360 versus 352 at K50. Every point is the original O@100 plus facet streams at K; both arms reach 462 at K100. This is headroom for ordering and fusion, not a final-ranking win."
            ),
        },
        {
            "id": "candidate_chart_block",
            "type": "chart",
            "chartId": "candidate_union_retention",
            "layout": "full",
        },
        {
            "id": "retention_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "facet_retention",
            "body": (
                "**Per-facet retention shows where each cutoff changes the evidence pool.** "
                "The table reports relevant documents retained independently within every "
                "frozen facet stream for corrected C0 and BF100 at K20, K50, and K100; it "
                "does not add counts across overlapping facets."
            ),
        },
        {
            "id": "retention_block",
            "type": "table",
            "tableId": "facet_retention",
            "layout": "full",
        },
        {
            "id": "fusion_heading",
            "type": "markdown",
            "layout": "full",
            "sourceId": "decision",
            "body": (
                "## Fusion blocks every novel relevant promotion\n\n"
                "**The separation between raw union, local reranking, and final ranking identifies the failure stage.** Of 220 relevant documents outside corrected C0's final top 100, MiniLM places 22 into the local top-20 promotion window (BF best facet rank ≤20 and C0 best facet rank >20); all 22 are fusion-blocked, leaving zero final novel relevant documents. The implication is to change fusion before collecting new candidates."
            ),
        },
        {
            "id": "fusion_chart_block",
            "type": "chart",
            "chartId": "promotion_funnel",
            "layout": "full",
        },
        {
            "id": "system_heading",
            "type": "markdown",
            "layout": "full",
            "sourceId": "decision",
            "body": (
                "## Final effectiveness is flat on coverage and mixed by topic\n\n"
                f"**BF100 versus corrected C0 is +0.0025617 nDCG@10 with 0 change in Recall@100 and graded Recall@100.** Topic nDCG@10 deltas are +0.0097987 (200), -0.0216612 (225), +0.0221092 (707), and 0 (897). BF100 has two relevant documents beyond legacy R1, but corrected C0 has those same two; that difference comes from the fusion correction, not MiniLM."
            ),
        },
        {
            "id": "topic_delta_chart_block",
            "type": "chart",
            "chartId": "topic_ndcg_delta",
            "layout": "full",
        },
        {
            "id": "primary_metrics_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "systems",
            "body": "**Primary values remain available at macro and topic grain for every preregistered system.** nDCG@10 describes top-rank ordering; Recall@100 and graded Recall@100 describe final coverage.",
        },
        {"id": "primary_metrics_block", "type": "table", "tableId": "system_primary", "layout": "full"},
        {
            "id": "diagnostic_metrics_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "systems",
            "body": "**Diagnostic metrics show top-10 precision, relevant counts, judgment coverage, and the best possible nDCG@10 available inside each frozen candidate cutoff.** They are diagnostics, not promotion criteria.",
        },
        {"id": "diagnostic_metrics_block", "type": "table", "tableId": "system_diagnostics", "layout": "full"},
        {
            "id": "gain_loss_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "gains_losses",
            "body": "**Relevant gains and losses are reported explicitly, including zeros.** BF100 gains no final relevant documents and loses none versus corrected C0; the legacy comparison is shown separately so the fusion correction cannot be misattributed.",
        },
        {"id": "gain_loss_block", "type": "table", "tableId": "gain_loss", "layout": "full"},
        {
            "id": "scope_heading",
            "type": "markdown",
            "layout": "full",
            "sourceId": "manifest",
            "body": (
                "## Full narratives and exact query scope\n\n"
                "**All four original narratives and all 31 frozen query streams are reproduced below.** The experiment contains four originals and 27 facets, each with 100 candidates (3,100 rows total)."
            ),
        },
        {"id": "narratives_block", "type": "table", "tableId": "narratives", "layout": "full"},
        {"id": "queries_block", "type": "table", "tableId": "query_streams", "layout": "full"},
        {
            "id": "scope_isolation_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "ranking_freeze",
            "body": "**MiniLM reranks facet streams only; original-query order is unchanged in every BF arm.**",
        },
        {
            "id": "facet_review_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "review_metrics",
            "body": "**Per-facet counts keep heterogeneous outcomes visible.** Each row uses two reviewed memberships for one arm/facet pairing; aggregate rates should not be read as uniform improvement across every facet.",
        },
        {"id": "facet_review_block", "type": "table", "tableId": "facet_review", "layout": "full"},
        {
            "id": "representative_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "representative_provenance",
            "body": "**Bounded promoted, demoted, gained, and lost examples connect the counts to facet-specific MiniLM passages.** The offline v2 provenance selects the qualifying facet event and highest-logit saved window without reopening qrels. Passages are truncated to 700 characters; empty gained/lost classes are retained rather than hidden.",
        },
        {"id": "representatives_block", "type": "table", "tableId": "representatives", "layout": "full"},
        {
            "id": "definition_population",
            "type": "markdown",
            "layout": "full",
            "sourceId": "manifest",
            "body": (
                "## Scope, data, metrics, and experimental design\n\n"
                "**Population.** The descriptive pilot contains topics 200, 225, 707, and 897; protected topics are excluded."
            ),
        },
        {
            "id": "definition_baseline",
            "type": "markdown",
            "layout": "full",
            "sourceId": "decision",
            "body": "**Baseline.** Corrected `C0_TOPIC_LOCAL` is the primary B comparison. `R1_LEGACY` is diagnostic only, and its difference from corrected C0 is audited separately.",
        },
        {
            "id": "definition_relevance_review",
            "type": "markdown",
            "layout": "full",
            "sourceId": "qrels_access_receipt",
            "body": "**Relevance and review.** Topic relevance comes from the one-time authorized Umbrela projection; raw qrels rows are not included. Facet relevance comes from a qrels-blind, system-masked review: 101 unique passages map to 108 arm/facet memberships. Seven shared passages are attributed to both arm memberships, preserving the preregistered denominators.",
        },
        {
            "id": "definition_metrics",
            "type": "markdown",
            "layout": "full",
            "sourceId": "decision",
            "body": "**Metrics.** Raw-union relevant documents count unique topic-qualified relevant documents from the original O@100 plus facet streams at K20/K50/K100. Pre-fusion promoted novel counts documents moved into the local top-20 promotion window (BF best facet rank ≤20 and C0 best facet rank >20) that are absent from corrected C0's final top 100. Final novel counts relevant documents in BF100's final top 100 but absent from corrected C0. Recall@100 is binary relevant coverage; graded Recall@100 weights relevance grades; nDCG@10 measures graded ordering in the first ten ranks.",
        },
        {
            "id": "definition_isolation",
            "type": "markdown",
            "layout": "full",
            "sourceId": "ranking_freeze",
            "body": "**Isolation.** BF changes facet-local ordering only. Raw MiniLM logits stay within query streams; RRF consumes ranks. Candidate generation, local reranking, fusion, and final ranking are measured separately.",
        },
        {
            "id": "firewall_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "qrels_access_receipt",
            "body": (
                "## Evaluation firewall stayed closed until both freezes\n\n"
                "**The exact four-topic projection was opened once, only after rankings and "
                "the blinded review were frozen.** The one-time qrels access receipt binds "
                "the same ranking and review freeze hashes as every evaluation artifact. "
                "The table also preserves the 101-item/108-membership review audit and the "
                "seven shared memberships attributed to both arms."
            ),
        },
        {
            "id": "firewall_block",
            "type": "table",
            "tableId": "evaluation_firewall",
            "layout": "full",
        },
        {
            "id": "model_heading",
            "type": "markdown",
            "layout": "full",
            "sourceId": "preflight",
            "body": (
                "## Model and bounded-window execution stayed within the preregistered envelope\n\n"
                "**The actual cross-encoder is `cross-encoder/ms-marco-MiniLM-L6-v2` at the pinned revision shown below, materialized through a six-file safe-file allowlist.** A tokenizer-only preflight created bounded windows before approved local ROCm scoring. Thirty long documents hit the 32-window cap; minimum token coverage was 29.52%, while median and p95 coverage were 100%."
            ),
        },
        {"id": "model_block", "type": "table", "tableId": "model_preflight", "layout": "full"},
        {
            "id": "scoring_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "scoring_receipt",
            "body": "**Approved scoring completed all 14,720 planned windows with no failed windows and 14,459 unique raw-logit scores.**",
        },
        {"id": "scoring_block", "type": "table", "tableId": "scoring_runtime", "layout": "full"},
        {
            "id": "ranking_freeze_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "ranking_freeze",
            "body": "**No retrieval, hosted inference, or qrels access occurred during ranking freeze.**",
        },
        {
            "id": "benchmark_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "benchmark",
            "body": "**The preregistered bounded benchmark ran on the local ROCm device before full scoring approval.** Its throughput and resource envelope are retained below as runtime evidence, not effectiveness evidence.",
        },
        {"id": "benchmark_block", "type": "table", "tableId": "benchmark_runtime", "layout": "full"},
        {
            "id": "limitations_title",
            "type": "markdown",
            "layout": "full",
            "body": "## Limitations and robustness",
        },
        {
            "id": "limitation_pilot",
            "type": "markdown",
            "layout": "full",
            "sourceId": "manifest",
            "body": "- **Descriptive pilot only.** Four topics cannot support an inferential or production-generalization claim.",
        },
        {
            "id": "limitation_judgments",
            "type": "markdown",
            "layout": "full",
            "sourceId": "qrels_access_receipt",
            "body": "- **Pooled judgments are incomplete.** Umbrela judgments are pooled and non-exhaustive; unjudged documents are not proof of irrelevance.",
        },
        {
            "id": "limitation_review",
            "type": "markdown",
            "layout": "full",
            "sourceId": "qrels_access_receipt",
            "body": "- **Review estimates are small and dependent.** The 101 unique passages produce 108 arm/facet memberships, with seven shared passages attributed to both arms. No confidence interval or significance test is claimed.",
        },
        {
            "id": "limitation_window",
            "type": "markdown",
            "layout": "full",
            "sourceId": "preflight",
            "body": "- **Window capping is a sensitivity risk.** Thirty long documents were sampled at a maximum of 32 windows; minimum retained token coverage was 29.52%.",
        },
        {
            "id": "limitation_fusion",
            "type": "markdown",
            "layout": "full",
            "sourceId": "decision",
            "body": "- **The fusion diagnosis is configuration-specific.** The evidence isolates the current family-balanced RRF, not every possible rank-fusion design.",
        },
        {
            "id": "recommendation",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Recommended next step\n\n"
                "1. Modify the topic-local fusion so facet-local promotions can survive without changing the frozen original-query order.\n"
                "2. Reuse the saved rankings and scores to test fusion-only alternatives; do not run new retrieval or inference for that diagnosis.\n"
                "3. Require positive final novel relevant coverage versus the identical corrected C0 fusion before permitting Stage A.\n"
                "4. Preserve nDCG@10 and per-topic losses as guardrails while optimizing coverage promotion."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "- Which fusion rule retains the 22 pre-fusion relevant promotions without harming topic 225?\n"
                "- Are gains robust to excluding the 30 capped documents or increasing their bounded window coverage in a separately approved run?\n"
                "- Do facet families need topic-specific weights, fewer correlated streams, or explicit novelty credit?\n"
                "- After fusion is fixed, does the same pattern replicate on untouched topics under a new preregistered evaluation?"
            ),
        },
    ]

    charts = [
        {
            "id": "facet_review_rates",
            "title": "Blinded facet-review rates by label and arm",
            "subtitle": "Two-arm comparison; 54 arm/facet memberships per arm, rates in percent.",
            "type": "bar",
            "dataset": "review_rate_rows",
            "sourceId": "review_metrics",
            "valueFormat": "number",
            "layout": "full",
            "encodings": {
                "x": {"field": "label", "type": "nominal", "label": "Review label"},
                "y": {"field": "rate_percent", "type": "quantitative", "label": "Rate (%)", "format": "number"},
                "color": {"field": "arm", "type": "nominal", "label": "Arm"},
                "label": {"field": "rate_percent", "type": "quantitative", "label": "Rate (%)"},
                "tooltip": [
                    {"field": "count", "type": "quantitative", "label": "Count"},
                    {"field": "denominator", "type": "quantitative", "label": "Membership denominator"},
                ],
            },
        },
        {
            "id": "candidate_union_retention",
            "title": "Relevant raw-union documents at K20, K50, and K100",
            "subtitle": "Unique relevant documents in O@100 plus facet streams at K; C0 versus BF100.",
            "type": "bar",
            "dataset": "candidate_union_rows",
            "sourceId": "raw_union",
            "valueFormat": "number",
            "layout": "full",
            "encodings": {
                "x": {"field": "depth", "type": "nominal", "label": "Facet-stream depth"},
                "y": {"field": "relevant_documents", "type": "quantitative", "label": "Relevant documents", "format": "number"},
                "color": {"field": "arm", "type": "nominal", "label": "Arm"},
                "label": {"field": "relevant_documents", "type": "quantitative", "label": "Relevant documents"},
                "tooltip": [
                    {"field": "unique_candidate_documents", "type": "quantitative", "label": "Unique candidate documents"},
                    {"field": "macro_recall", "type": "quantitative", "label": "Macro recall"},
                    {"field": "macro_graded_recall", "type": "quantitative", "label": "Macro graded recall"},
                ],
            },
        },
        {
            "id": "promotion_funnel",
            "title": "Relevant-document survival from headroom to final ranking",
            "subtitle": "Raw-union headroom, pre-fusion promotion, and final novelty versus corrected C0.",
            "type": "bar",
            "dataset": "stage_set_rows",
            "sourceId": "decision",
            "valueFormat": "number",
            "layout": "full",
            "encodings": {
                "x": {"field": "stage", "type": "nominal", "label": "Stage"},
                "y": {"field": "relevant_documents", "type": "quantitative", "label": "Relevant documents", "format": "number"},
                "label": {"field": "relevant_documents", "type": "quantitative", "label": "Relevant documents"},
                "tooltip": [{"field": "document_set_sha256", "type": "nominal", "label": "Set SHA-256"}],
            },
        },
        {
            "id": "topic_ndcg_delta",
            "title": "Per-topic nDCG@10 delta, BF100 minus corrected C0",
            "subtitle": "Signed ordering change across the four descriptive pilot topics; zero is no change.",
            "type": "bar",
            "dataset": "topic_ndcg_rows",
            "sourceId": "decision",
            "valueFormat": "number",
            "layout": "full",
            "encodings": {
                "x": {"field": "topic_id", "type": "nominal", "label": "Topic"},
                "y": {"field": "ndcg_delta", "type": "quantitative", "label": "nDCG@10 delta", "format": "number"},
                "label": {"field": "ndcg_delta", "type": "quantitative", "label": "Signed delta"},
                "tooltip": [
                    {"field": "c0_ndcg_at_10", "type": "quantitative", "label": "C0 nDCG@10"},
                    {"field": "bf100_ndcg_at_10", "type": "quantitative", "label": "BF100 nDCG@10"},
                    {"field": "recall_delta", "type": "quantitative", "label": "Recall@100 delta"},
                ],
            },
        },
    ]

    tables = [
        _table("narratives", "Full narrative by topic", "All four evaluated pilot narratives, reproduced without truncation.", "narrative_rows", "manifest", (("topic_id", "Topic", "text"), ("facet_count", "Facet streams", "number"), ("full_narrative", "Full narrative", "text")), "topic_id"),
        _table("query_streams", "Exact original and facet query streams", "Thirty-one frozen streams: four originals and 27 facets, each with 100 candidates.", "query_rows", "manifest", (("topic_id", "Topic", "text"), ("family", "Family", "text"), ("variant", "Facet or variant", "text"), ("full_query", "Full query", "text"), ("candidate_rows", "Candidate rows", "number")), "topic_id", density="dense"),
        _table("system_primary", "Primary system effectiveness metrics", "Macro and per-topic nDCG@10, graded Recall@100, and Recall@100 for every evaluated system.", "system_rows", "systems", (("row_label", "System and scope", "text"), ("ndcg_at_10", "nDCG@10", "number"), ("graded_recall_at_100", "Graded Recall@100", "number"), ("recall_at_100", "Recall@100", "number")), "row_label", density="dense"),
        _table("system_diagnostics", "Diagnostic system metrics", "Top-10 precision, relevant counts, judgment coverage, and candidate-oracle ordering diagnostics.", "system_rows", "systems", (("row_label", "System and scope", "text"), ("precision_at_10", "P@10", "number"), ("relevant_count_at_10", "Relevant@10", "number"), ("relevant_count_at_100", "Relevant@100", "number"), ("judged_rate_at_10", "Judged rate@10", "number"), ("judged_rate_at_100", "Judged rate@100", "number"), ("oracle_ndcg_at_10_from_top50", "Oracle nDCG@10 from top 50", "number"), ("oracle_ndcg_at_10_from_top100", "Oracle nDCG@10 from top 100", "number")), "row_label", density="dense"),
        _table("gain_loss", "Relevant-document gains and losses", "BF100 compared separately with corrected C0 and legacy R1 at aggregate and topic grain.", "gain_loss_rows", "gains_losses", (("row_label", "Comparison", "text"), ("baseline_relevant", "Baseline relevant", "number"), ("candidate_relevant", "BF100 relevant", "number"), ("gained", "Gained", "number"), ("lost", "Lost", "number"), ("net_change", "Net change", "number")), "row_label"),
        _table("facet_review", "Per-facet blinded review evidence", "Two memberships per arm/facet cell; rates and flags remain visible for all 27 facets.", "facet_review_rows", "review_metrics", (("topic_id", "Topic", "text"), ("facet", "Facet or variant", "text"), ("arm", "Arm", "text"), ("direct_answer", "Direct answers", "number"), ("direct_answer_percent", "Direct answer (%)", "number"), ("partial_or_related", "Partial/related", "number"), ("irrelevant", "Irrelevant", "number"), ("wrong_domain", "Wrong domain", "number"), ("low_quality", "Low quality", "number")), "topic_id", density="dense"),
        _table("facet_retention", "Per-facet relevant retention by cutoff", "Independent relevant-document counts for all 27 facets in corrected C0 and BF100.", "facet_retention_rows", "facet_retention", (("topic_id", "Topic", "text"), ("facet", "Facet or variant", "text"), ("arm", "Arm", "text"), ("relevant_at_k20", "Relevant at K20", "number"), ("relevant_at_k50", "Relevant at K50", "number"), ("relevant_at_k100", "Relevant at K100", "number")), "topic_id", density="dense"),
        _table("representatives", "Bounded representative passage evidence", "Up to three saved examples per evidence class with the exact facet event and MiniLM window; passage text capped at 700 characters.", "representative_rows", "representative_provenance", (("evidence_class", "Evidence class", "text"), ("topic_id", "Topic", "text"), ("document_id", "Document", "text"), ("facet", "Qualifying facet", "text"), ("c0_facet_rank", "C0 facet rank", "number"), ("bf_facet_rank", "BF facet rank", "number"), ("passage", "Passage excerpt", "text"), ("provenance", "Frozen provenance", "text")), "evidence_class"),
        _table("model_preflight", "Pinned model and capped-window coverage", "Tokenizer-only preflight evidence for the actual model identity, safe files, and bounded coverage.", "model_rows", "preflight", (("model", "Model", "text"), ("revision", "Revision", "text"), ("safe_files", "Safe-file allowlist", "text"), ("window_cap", "Window cap per document", "number"), ("capped_documents", "Capped documents", "number"), ("coverage_min_percent", "Minimum coverage (%)", "number"), ("coverage_p95_percent", "p95 coverage (%)", "number")), "model"),
        _table("scoring_runtime", "Completed local scoring", "Authenticated raw-logit scoring receipt; all planned windows completed.", "scoring_rows", "scoring_receipt", (("model", "Model", "text"), ("revision", "Revision", "text"), ("dtype", "Inference dtype", "text"), ("planned_windows", "Planned windows", "number"), ("completed_windows", "Completed windows", "number"), ("unique_scores", "Unique scores", "number"), ("failed_windows", "Failed windows", "number")), "model"),
        _table("benchmark_runtime", "ROCm benchmark evidence", "Bounded benchmark used for full-run approval and runtime projection.", "benchmark_rows", "benchmark", (("backend", "Backend", "text"), ("device", "Device", "text"), ("pairs_per_second", "Median pairs/s", "number"), ("peak_device_memory_mb", "Peak device memory (MB)", "number"), ("projected_wall_seconds", "Projected full wall time (s)", "number")), "backend"),
        _table("evaluation_firewall", "Evaluation firewall and blinded-review audit", "Path-independent git-common-dir qrels consumption and pre-qrels ranking/review freeze evidence; the legacy adjacent marker is retained but not trusted.", "firewall_rows", "qrels_access_receipt", (("qrels_access_status", "Qrels access status", "text"), ("qrels_access_count", "Access count", "number"), ("approval_scoped_consumption_registry", "Approval consumed globally", "boolean"), ("registry_namespace", "Trusted guard namespace", "text"), ("legacy_marker_trusted", "Legacy marker trusted", "boolean"), ("ranking_frozen_before_qrels", "Ranking frozen first", "boolean"), ("review_frozen_before_qrels", "Review frozen first", "boolean"), ("review_unique_items", "Unique review items", "number"), ("review_memberships", "Arm/facet memberships", "number"), ("shared_items_attributed_to_both_arms", "Shared memberships", "number")), "qrels_access_status"),
    ]

    sources = [_source(key, artifact_bytes) for key in SOURCE_PATHS]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Technical four-topic facet-local MiniLM diagnostic report.",
            "charts": charts,
            "tables": tables,
            "sources": [
                {"id": source["id"], "label": source["label"], "path": source["path"]}
                for source in sources
            ],
            "blocks": blocks,
        },
        "snapshot": {"version": 1, "status": "ready", "datasets": rows},
        "sources": sources,
    }


def _load(path: Path, label: str) -> tuple[Mapping[str, object], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable or invalid JSON") from exc
    return _object(payload, label), raw


def _write_create_only(path: Path, data: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise FileExistsError(f"create-only {label} already exists: {path}") from exc


def render_html_create_only(
    *,
    artifact_path: Path,
    output_path: Path,
    renderer_path: Path,
    runner: Callable[[Sequence[str]], object] | None = None,
) -> None:
    """Render to a private file and atomically publish without replacement."""

    if output_path.exists():
        raise FileExistsError(f"create-only HTML report already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    command = [
        "node",
        str(renderer_path),
        "--input",
        str(artifact_path),
        "--output",
        str(temporary_path),
    ]
    try:
        if runner is None:
            subprocess.run(command, check=True)
        else:
            runner(command)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError("portable renderer did not produce a non-empty HTML file")
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"create-only HTML report already exists: {output_path}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--renderer", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.html_output is None) != (args.renderer is None):
        raise ValueError("--html-output and --renderer must be supplied together")
    if args.html_output is not None and args.html_output.exists():
        raise FileExistsError(
            f"create-only HTML report already exists: {args.html_output}"
        )
    root = args.scoring.parent
    paths = {
        "manifest": args.manifest,
        "preflight": args.preflight / "preflight.json",
        "scoring_receipt": args.scoring / "scoring_receipt.json",
        "benchmark": root / "benchmark_v1/benchmark_telemetry.json",
        "model_download_approval": root / "approvals/model_download_v1.json",
        "benchmark_approval": root / "approvals/benchmark_v1.json",
        "full_scoring_approval": root / "approvals/full_scoring_v1.json",
        "ranking_freeze": args.freeze / "freeze.json",
        "legacy_corrected_diff": args.freeze / "legacy_corrected_diff.json",
        "review_freeze": args.review / "review_freeze.json",
        "review_create_receipt": args.review / "create_receipt.json",
        "raw_union": args.evaluation / "raw_union.json",
        "prefusion": args.evaluation / "prefusion.json",
        "facet_retention": args.evaluation / "facet_retention.json",
        "systems": args.evaluation / "systems.json",
        "gains_losses": args.evaluation / "gains_losses.json",
        "review_metrics": args.evaluation / "review_metrics.json",
        "representatives": args.evaluation / "representatives.json",
        "representative_provenance": root
        / "derived_v2/representative_provenance_v2.json",
        "decision": args.evaluation / "decision.json",
        "qrels_access_approval": root / "approvals/qrels_access_v1.json",
        "qrels_consumption_registry": root
        / "approvals/qrels_consumption_registry_v2.json",
        "qrels_consumption_legacy_marker": root
        / "approvals/qrels_access_v1.json.consumed.json",
        "qrels_access_receipt": args.evaluation / "qrels_access_receipt.json",
    }
    for key, path in paths.items():
        expected = REPO_ROOT / SOURCE_PATHS[key]
        if path.resolve() != expected.resolve():
            raise ValueError(
                f"noncanonical {key} path: expected {expected}, received {path}"
            )
    from trec_rag.facet_local_minilm_evaluate import (
        qrels_consumption_registry_path,
    )

    trusted_registry_path = qrels_consumption_registry_path(
        paths["qrels_access_approval"]
    )
    try:
        trusted_registry_bytes = trusted_registry_path.read_bytes()
        mirror_registry_bytes = paths["qrels_consumption_registry"].read_bytes()
    except OSError as exc:
        raise ValueError("trusted qrels consumption registry is unavailable") from exc
    if trusted_registry_bytes != mirror_registry_bytes:
        raise ValueError("qrels consumption registry mirror differs from trusted state")
    payloads: dict[str, object] = {}
    raw: dict[str, bytes] = {}
    for key, path in paths.items():
        payloads[key], raw[key] = _load(path, key)
    artifact = build_artifact(
        artifacts=payloads,
        artifact_bytes=raw,
        decision=_object(payloads["decision"], "decision"),
    )
    _write_create_only(
        args.output,
        (json.dumps(artifact, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        "report source",
    )
    if args.html_output is not None:
        render_html_create_only(
            artifact_path=args.output,
            output_path=args.html_output,
            renderer_path=args.renderer,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
