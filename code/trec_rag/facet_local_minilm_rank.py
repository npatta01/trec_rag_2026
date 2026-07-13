"""Offline aggregation and ranking freeze for the facet-local MiniLM pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

from .pipeline_models import RetrievedCandidate, jsonable
from .ranking import reciprocal_rank_fusion


_TOP4_WEIGHTS = (0.55, 0.25, 0.13, 0.07)
_MINIMUM_NEW_TOKENS = 128
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
ARM_NAMES = (
    "R1_LEGACY",
    "C0_TOPIC_LOCAL",
    "BF100_TOPIC_LOCAL",
    "BF50_TOPIC_LOCAL",
    "BF20_TOPIC_LOCAL",
    "BO100_TOPIC_LOCAL",
    "BB100_TOPIC_LOCAL",
    "BF100_MAXP_TOPIC_LOCAL",
    "BF100_LEGACY_FUSION",
)
PILOT_TOPIC_IDS = ("200", "225", "707", "897")
RETRIEVER_NAME = "pyserini_remote_raw_first_v1"
AUTHENTICATED_SCORING_RECEIPT_SHA256 = (
    "d8d3f86d16bd25d0f92c682f3707273b08279489636481b18f694e9dd673df00"
)


@dataclass(frozen=True)
class AuthenticatedInputs:
    manifest: Mapping[str, object]
    candidates: tuple[Mapping[str, object], ...]
    windows: tuple[Mapping[str, object], ...]
    scores: tuple[Mapping[str, object], ...]
    bindings: Mapping[str, object]


def _new_token_count(start: int, end: int, covered: list[tuple[int, int]]) -> int:
    overlaps = sorted(
        (max(start, left), min(end, right))
        for left, right in covered
        if min(end, right) > max(start, left)
    )
    covered_count = 0
    cursor = start
    for left, right in overlaps:
        if right <= cursor:
            continue
        covered_count += right - max(left, cursor)
        cursor = right
    return end - start - covered_count


def _top4_selection(
    windows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    ordered = sorted(
        windows,
        key=lambda row: (
            -float(row["score"]),
            int(row["document_start_token"]),
            str(row["window_id"]),
        ),
    )
    accepted: list[Mapping[str, object]] = []
    covered: list[tuple[int, int]] = []
    for row in ordered:
        score = float(row["score"])
        start = int(row["document_start_token"])
        end = int(row["document_end_token"])
        if not math.isfinite(score) or start < 0 or end <= start:
            raise ValueError("window score and token span must be valid")
        if accepted and _new_token_count(start, end, covered) < _MINIMUM_NEW_TOKENS:
            continue
        accepted.append(row)
        covered.append((start, end))
        if len(accepted) == len(_TOP4_WEIGHTS):
            break
    if not accepted:
        raise ValueError("top-four aggregation requires at least one scored window")
    return accepted


def aggregate_top4(windows: Sequence[Mapping[str, object]]) -> float:
    """Return the frozen top-four span-distinct aggregate for one document."""

    accepted = _top4_selection(windows)
    weights = _TOP4_WEIGHTS[: len(accepted)]
    denominator = math.fsum(weights)
    return math.fsum(
        weight * float(row["score"])
        for weight, row in zip(weights, accepted, strict=True)
    ) / denominator


def aggregate_maxp(windows: Sequence[Mapping[str, object]]) -> float:
    """Return the maximum finite raw logit for one query-document pair."""

    if not windows:
        raise ValueError("MaxP aggregation requires at least one scored window")
    scores = [float(row["score"]) for row in windows]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("window scores must be finite")
    return max(scores)


def rerank_stream(
    candidates: Sequence[Mapping[str, object]],
    windows: Sequence[Mapping[str, object]],
    *,
    aggregation: str,
) -> list[dict[str, object]]:
    """Rerank one stream using only its own query-specific MiniLM scores."""

    if aggregation not in {"top4", "maxp"}:
        raise ValueError("aggregation must be 'top4' or 'maxp'")
    if not candidates:
        raise ValueError("stream reranking requires candidates")
    candidate_docids = [str(row["document_id"]) for row in candidates]
    if len(set(candidate_docids)) != len(candidate_docids):
        raise ValueError("stream candidates must have distinct document IDs")
    window_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for window in windows:
        window_rows[str(window["document_id"])].append(window)
    if set(window_rows) != set(candidate_docids):
        raise ValueError("scored windows must cover exactly the stream candidates")

    reranked: list[dict[str, object]] = []
    for candidate in candidates:
        docid = str(candidate["document_id"])
        document_windows = window_rows[docid]
        if aggregation == "top4":
            selected = _top4_selection(document_windows)
            score = aggregate_top4(document_windows)
        else:
            selected = [
                min(
                    document_windows,
                    key=lambda row: (
                        -float(row["score"]),
                        int(row["document_start_token"]),
                        str(row["window_id"]),
                    ),
                )
            ]
            score = aggregate_maxp(document_windows)
        prior_rank = int(candidate["rank"])
        row = dict(candidate)
        row.update(
            {
                "aggregation": aggregation,
                "passage": str(selected[0].get("window_text", "")),
                "prior_rank": prior_rank,
                "score": score,
                "selected_windows": [
                    {
                        "document_end_token": int(item["document_end_token"]),
                        "document_start_token": int(item["document_start_token"]),
                        "score": float(item["score"]),
                        "window_id": str(item["window_id"]),
                        "window_text": str(item.get("window_text", "")),
                    }
                    for item in selected
                ],
            }
        )
        reranked.append(row)

    ordered = sorted(
        reranked,
        key=lambda row: (
            -float(row["score"]),
            int(row["prior_rank"]),
            str(row["document_id"]),
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return ordered


def build_weight_tables(
    streams: Sequence[Mapping[str, object]],
    *,
    retriever_name: str,
) -> tuple[
    dict[tuple[str, str], Fraction],
    dict[tuple[str, str, str], Fraction],
]:
    """Reconstruct legacy global-key v1 and corrected topic-local v2 weights."""

    if not retriever_name:
        raise ValueError("retriever name must be non-empty")
    indexed: dict[tuple[str, str], str] = {}
    for stream in streams:
        topic_id = str(stream["topic_id"])
        variant = str(stream["variant"])
        family = str(stream["family"])
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if not topic_id or not variant or family not in {"original", "facet"}:
            raise ValueError("stream identity and family must be valid")
        key = (topic_id, variant)
        if key in indexed:
            raise ValueError(f"duplicate stream identity: {key!r}")
        indexed[key] = family

    topics = sorted({topic_id for topic_id, _variant in indexed})
    legacy: dict[tuple[str, str], Fraction] = {}
    corrected: dict[tuple[str, str, str], Fraction] = {}
    for topic_id in topics:
        topic_streams = sorted(
            (variant, family)
            for (topic, variant), family in indexed.items()
            if topic == topic_id
        )
        originals = [variant for variant, family in topic_streams if family == "original"]
        facets = [variant for variant, family in topic_streams if family == "facet"]
        if len(originals) != 1 or not facets:
            raise ValueError(
                f"topic {topic_id} requires exactly one original and at least one facet"
            )
        original_weight = Fraction(1, 2)
        facet_weight = Fraction(1, 2 * len(facets))
        corrected[(topic_id, originals[0], retriever_name)] = original_weight
        legacy[(originals[0], retriever_name)] = original_weight
        for variant in facets:
            corrected[(topic_id, variant, retriever_name)] = facet_weight
            legacy[(variant, retriever_name)] = facet_weight
    return legacy, corrected


def build_arm_streams(
    streams: Mapping[
        tuple[str, str, str],
        Mapping[str, object],
    ],
) -> dict[str, dict[tuple[str, str, str], tuple[object, ...]]]:
    """Apply the frozen offline reranking and retention matrix to 31 streams."""

    if not streams:
        raise ValueError("arm construction requires stream rankings")
    result = {
        arm: {}
        for arm in ARM_NAMES
        if arm != "R1_LEGACY"
    }
    families_by_topic: dict[str, list[str]] = defaultdict(list)
    for identity in sorted(streams):
        topic_id, _variant, retriever = identity
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if not retriever:
            raise ValueError("stream retriever identity must be non-empty")
        record = streams[identity]
        family = str(record.get("family"))
        if family not in {"original", "facet"}:
            raise ValueError("stream family must be original or facet")
        families_by_topic[topic_id].append(family)
        bm25 = tuple(record.get("bm25", ()))
        top4 = tuple(record.get("top4", ()))
        maxp = tuple(record.get("maxp", ()))
        if not bm25 or len(bm25) != len(top4) or len(bm25) != len(maxp):
            raise ValueError("every stream requires aligned BM25, top-four, and MaxP rows")

        result["C0_TOPIC_LOCAL"][identity] = bm25
        result["BF100_TOPIC_LOCAL"][identity] = bm25 if family == "original" else top4
        result["BF50_TOPIC_LOCAL"][identity] = (
            bm25 if family == "original" else top4[:50]
        )
        result["BF20_TOPIC_LOCAL"][identity] = (
            bm25 if family == "original" else top4[:20]
        )
        result["BO100_TOPIC_LOCAL"][identity] = top4 if family == "original" else bm25
        result["BB100_TOPIC_LOCAL"][identity] = top4
        result["BF100_MAXP_TOPIC_LOCAL"][identity] = (
            bm25 if family == "original" else maxp
        )
        result["BF100_LEGACY_FUSION"][identity] = (
            bm25 if family == "original" else top4
        )

    for topic_id, families in families_by_topic.items():
        if families.count("original") != 1 or families.count("facet") < 1:
            raise ValueError(
                f"topic {topic_id} requires exactly one original and at least one facet"
            )
    return result


def _stream_weight(
    identity: tuple[str, str, str],
    weights: Mapping[tuple[str, ...], Fraction | float],
) -> float:
    topic_id, variant, retriever = identity
    value = weights.get((topic_id, variant, retriever))
    if value is None:
        value = weights.get((variant, retriever))
    if value is None:
        raise ValueError(f"missing RRF weight for stream {identity!r}")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("RRF weights must be finite and positive")
    return weight


def fuse_streams(
    streams: Mapping[tuple[str, str, str], Sequence[Mapping[str, object]]],
    weights: Mapping[tuple[str, ...], Fraction | float],
    *,
    k: int = 60,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Fuse topic-qualified stream ranks without cross-stream score comparison."""

    if k != 60 or limit != 100:
        raise ValueError("the frozen RRF definition requires k=60 and limit=100")
    contributions_by_doc: dict[
        tuple[str, str], list[tuple[tuple[str, str, str], Mapping[str, object], float]]
    ] = defaultdict(list)
    for identity in sorted(streams):
        topic_id, variant, retriever = identity
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        weight = _stream_weight(identity, weights)
        seen: set[str] = set()
        for row in sorted(
            streams[identity],
            key=lambda item: (int(item["rank"]), str(item["document_id"])),
        ):
            if (
                str(row["topic_id"]) != topic_id
                or str(row["variant"]) != variant
            ):
                raise ValueError("stream row identity differs from its topic-qualified key")
            docid = str(row["document_id"])
            rank = int(row["rank"])
            score = float(row["score"])
            if not docid or rank < 1 or not math.isfinite(score):
                raise ValueError("stream rows require valid IDs, ranks, and scores")
            if docid in seen:
                raise ValueError("stream rows must be deduplicated by document ID")
            seen.add(docid)
            contributions_by_doc[(topic_id, docid)].append(
                (identity, row, weight / (k + rank))
            )

    by_topic: dict[
        str, list[tuple[str, float, int, str, list[dict[str, object]]]]
    ] = defaultdict(list)
    for (topic_id, docid), contributions in contributions_by_doc.items():
        ordered = sorted(contributions, key=lambda item: item[0])
        representative_identity, representative, _ = min(
            ordered,
            key=lambda item: (int(item[1]["rank"]), item[0]),
        )
        del representative_identity
        provenance: list[dict[str, object]] = []
        for (stream_topic, variant, retriever), row, contribution in ordered:
            if stream_topic != topic_id:
                raise AssertionError("cross-topic fusion contribution")
            provenance.append(
                {
                    "aggregation": str(row.get("aggregation", "unknown")),
                    "passage": str(row.get("passage", "")),
                    "prior_rank": int(row.get("prior_rank", row["rank"])),
                    "query_text": str(row.get("query", "")),
                    "ranker": "reciprocal_rank_fusion",
                    "retrieval_score": float(row.get("source_score", row["score"])),
                    "retriever_name": retriever,
                    "rrf_contribution": contribution,
                    "rrf_k": k,
                    "rrf_weight": _stream_weight(
                        (topic_id, variant, retriever), weights
                    ),
                    "selected_windows": list(row.get("selected_windows", [])),
                    "source_rank": int(row["rank"]),
                    "source_score": float(row["score"]),
                    "variant_name": variant,
                }
            )
        by_topic[topic_id].append(
            (
                docid,
                math.fsum(item[2] for item in ordered),
                min(int(item[1]["rank"]) for item in ordered),
                str(representative.get("passage", representative.get("text", ""))),
                provenance,
            )
        )

    result: list[dict[str, object]] = []
    for topic_id in sorted(by_topic):
        ranked = sorted(
            by_topic[topic_id],
            key=lambda item: (-item[1], item[2], item[0]),
        )[:limit]
        for rank, (docid, score, _source_rank, text, provenance) in enumerate(
            ranked, start=1
        ):
            result.append(
                {
                    "docid": docid,
                    "provenance": provenance,
                    "rank": rank,
                    "score": score,
                    "text": text,
                    "topic_id": topic_id,
                }
            )
    return result


def canonical_ranking_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    """Hash ranking meaning with the historical compact-array convention."""

    payload = (
        json.dumps(
            list(rows),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reconstruct_legacy_ranking(
    candidate_rows: Sequence[Mapping[str, object]],
    legacy_weights: Mapping[tuple[str, str], Fraction | float],
) -> list[dict[str, object]]:
    """Reproduce historical R1 with the exact topic-free v1 weight mapping."""

    retrievers = {retriever for _variant, retriever in legacy_weights}
    if len(retrievers) != 1:
        raise ValueError("legacy reconstruction requires one retriever identity")
    retriever = next(iter(retrievers))
    candidates: list[RetrievedCandidate] = []
    for raw in candidate_rows:
        topic_id = str(raw["topic_id"])
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        variant = str(raw["variant"])
        if (variant, retriever) not in legacy_weights:
            raise ValueError("candidate stream is absent from the legacy v1 table")
        candidates.append(
            RetrievedCandidate(
                topic_id=topic_id,
                variant_name=variant,
                retriever_name=retriever,
                query_text=str(raw["query"]),
                docid=str(raw["document_id"]),
                rank=int(raw["rank"]),
                score=float(raw["source_score"]),
                text=str(raw["text"]),
            )
        )
    ranked = reciprocal_rank_fusion(
        candidates,
        k=60,
        stream_weights={key: float(value) for key, value in legacy_weights.items()},
        limit=100,
    )
    return [jsonable(row) for row in ranked]


def build_control_diff(
    legacy_rows: Sequence[Mapping[str, object]],
    corrected_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the frozen qrels-free per-topic rank/set correction audit."""

    def index(
        rows: Sequence[Mapping[str, object]], label: str
    ) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = defaultdict(dict)
        for row in rows:
            topic_id = str(row["topic_id"])
            if topic_id in PROTECTED_TOPIC_IDS:
                raise ValueError(f"protected topic {topic_id} is forbidden")
            docid = str(row["docid"])
            rank = int(row["rank"])
            if not docid or rank < 1 or docid in result[topic_id]:
                raise ValueError(f"{label} ranking contains invalid or duplicate rows")
            result[topic_id][docid] = rank
        return dict(result)

    legacy = index(legacy_rows, "legacy")
    corrected = index(corrected_rows, "corrected")
    if set(legacy) != set(corrected):
        raise ValueError("legacy and corrected controls must cover identical topics")
    topics: dict[str, object] = {}
    for topic_id in sorted(legacy):
        legacy_docids = set(legacy[topic_id])
        corrected_docids = set(corrected[topic_id])
        shared = legacy_docids & corrected_docids
        rank_changes = [
            {
                "corrected_rank": corrected[topic_id][docid],
                "document_id": docid,
                "legacy_rank": legacy[topic_id][docid],
                "rank_delta_corrected_minus_legacy": (
                    corrected[topic_id][docid] - legacy[topic_id][docid]
                ),
            }
            for docid in sorted(shared)
            if corrected[topic_id][docid] != legacy[topic_id][docid]
        ]
        topics[topic_id] = {
            "corrected_only": sorted(corrected_docids - legacy_docids),
            "legacy_only": sorted(legacy_docids - corrected_docids),
            "rank_changes": rank_changes,
            "shared_document_count": len(shared),
        }
    return {
        "qrels_opened": False,
        "schema_version": "facet-local-minilm-control-diff-v1",
        "topics": topics,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable or invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    try:
        with path.open("rb") as source:
            for line_number, raw in enumerate(source, start=1):
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError(f"{label} row {line_number} must be an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable or invalid JSONL") from exc
    return tuple(rows)


def _window_identity(row: Mapping[str, object]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["topic_id"]),
        str(row["variant"]),
        str(row["document_id"]),
        str(row["window_id"]),
        str(row["query_sha256"]),
        str(row["window_sha256"]),
    )


def load_authenticated_inputs(
    *,
    manifest_path: Path,
    preflight_dir: Path,
    scores_dir: Path,
) -> AuthenticatedInputs:
    """Load only the authenticated source snapshot, preflight, and saved scores."""

    manifest_path = Path(manifest_path)
    preflight_dir = Path(preflight_dir)
    scores_dir = Path(scores_dir)
    manifest = _read_json(manifest_path, "experiment manifest")
    preflight_path = preflight_dir / "preflight.json"
    windows_path = preflight_dir / "windows.jsonl"
    preflight = _read_json(preflight_path, "tokenizer preflight")
    scoring_receipt_path = scores_dir / "scoring_receipt.json"
    scoring_receipt = _read_json(scoring_receipt_path, "scoring receipt")
    terminal = _read_json(scores_dir / "run_terminal.json", "scoring terminal")
    ledger_path = scores_dir / "scoring_ledger.jsonl"

    manifest_sha = _sha256_file(manifest_path)
    preflight_sha = _sha256_file(preflight_path)
    windows_sha = _sha256_file(windows_path)
    scoring_receipt_sha = _sha256_file(scoring_receipt_path)
    ledger_sha = _sha256_file(ledger_path)
    if scoring_receipt_sha != AUTHENTICATED_SCORING_RECEIPT_SHA256:
        raise ValueError("scoring receipt differs from the authenticated Task 4 receipt")
    if (
        manifest.get("schema_version") != "facet-local-minilm-manifest-v1"
        or manifest.get("status") != "frozen_source_snapshot"
        or tuple(manifest.get("topic_ids", ())) != PILOT_TOPIC_IDS
        or set(manifest.get("protected_topic_ids", ())) != PROTECTED_TOPIC_IDS
        or manifest.get("stream_count") != 31
        or manifest.get("candidate_rows") != 3100
    ):
        raise ValueError("experiment manifest differs from the frozen Task 4 contract")
    if (
        preflight.get("schema_version") != "facet-local-minilm-preflight-v2"
        or preflight.get("status") != "tokenizer_only_preflight_complete"
        or preflight.get("manifest_sha256") != manifest_sha
        or preflight.get("source_candidates_sha256")
        != manifest.get("candidates_sha256")
        or preflight.get("source_candidate_rows") != 3100
        or preflight.get("windows_sha256") != windows_sha
        or preflight.get("qrels_path_supported") is not False
        or preflight.get("retrieval_path_supported") is not False
    ):
        raise ValueError("preflight bindings differ from the authenticated source")
    if (
        scoring_receipt.get("schema_version")
        != "facet-local-minilm-scoring-receipt-v2"
        or scoring_receipt.get("status") != "complete"
        or scoring_receipt.get("preflight_sha256") != preflight_sha
        or scoring_receipt.get("windows_sha256") != windows_sha
        or scoring_receipt.get("ledger_sha256") != ledger_sha
        or scoring_receipt.get("planned_window_count") != 14720
        or scoring_receipt.get("completed_window_count") != 14720
        or scoring_receipt.get("failed_window_count") != 0
        or scoring_receipt.get("pending_window_count") != 0
        or scoring_receipt.get("inference_dtype") != "float32"
        or scoring_receipt.get("score_representation") != "raw_logits"
    ):
        raise ValueError("scoring artifacts differ from the complete authenticated run")
    if (
        terminal.get("status") != "complete"
        or terminal.get("scoring_receipt_sha256") != scoring_receipt_sha
    ):
        raise ValueError("scoring terminal does not bind the authenticated receipt")

    source_dir_raw = preflight.get("source_dir")
    if not isinstance(source_dir_raw, str) or not source_dir_raw:
        raise ValueError("preflight lacks the authenticated source snapshot directory")
    source_dir = Path(source_dir_raw)
    candidate_path = source_dir / str(manifest["candidate_file"])
    source_receipt_path = source_dir / str(manifest["source_receipt_file"])
    source_receipt = _read_json(source_receipt_path, "source receipt")
    if (
        _sha256_file(source_receipt_path) != manifest.get("source_receipt_sha256")
        or preflight.get("source_receipt_sha256")
        != manifest.get("source_receipt_sha256")
        or _sha256_file(candidate_path) != manifest.get("candidates_sha256")
        or source_receipt.get("candidates_sha256") != manifest.get("candidates_sha256")
        or source_receipt.get("candidate_rows") != 3100
        or source_receipt.get("stream_count") != 31
        or source_receipt.get("status") != "frozen_source_snapshot"
    ):
        raise ValueError("source snapshot or receipt differs from the manifest")

    candidates = _read_jsonl(candidate_path, "source candidates")
    if len(candidates) != 3100:
        raise ValueError("source snapshot does not contain exactly 3,100 rows")
    observed_topics = {str(row.get("topic_id")) for row in candidates}
    protected = sorted(observed_topics & PROTECTED_TOPIC_IDS)
    if protected:
        raise ValueError(f"protected topic {protected[0]} is forbidden")
    if observed_topics != set(PILOT_TOPIC_IDS):
        raise ValueError("source candidates differ from the four pilot topics")

    windows = _read_jsonl(windows_path, "preflight windows")
    scores = _read_jsonl(ledger_path, "scoring ledger")
    if len(windows) != 14720 or len(scores) != 14720:
        raise ValueError("preflight/scoring rows differ from the exact window count")
    window_index = {_window_identity(row) for row in windows}
    score_index = {_window_identity(row) for row in scores}
    if (
        len(window_index) != len(windows)
        or len(score_index) != len(scores)
        or window_index != score_index
    ):
        raise ValueError("scoring ledger does not exactly cover the preflight windows")
    if any(str(row.get("topic_id")) in PROTECTED_TOPIC_IDS for row in (*windows, *scores)):
        raise ValueError("protected topic is forbidden in preflight or scoring artifacts")

    bindings: dict[str, object] = {
        "candidates_sha256": str(manifest["candidates_sha256"]),
        "inference_call_count": 0,
        "ledger_sha256": ledger_sha,
        "manifest_sha256": manifest_sha,
        "preflight_sha256": preflight_sha,
        "qrels_opened": False,
        "retrieval_call_count": 0,
        "scoring_receipt_sha256": scoring_receipt_sha,
        "source_receipt_sha256": str(manifest["source_receipt_sha256"]),
        "windows_sha256": windows_sha,
    }
    return AuthenticatedInputs(
        manifest=manifest,
        candidates=candidates,
        windows=windows,
        scores=scores,
        bindings=bindings,
    )


def build_stream_rankings(
    inputs: AuthenticatedInputs,
) -> dict[tuple[str, str, str], dict[str, object]]:
    """Materialize BM25, top-four, and MaxP orders from authenticated scores."""

    manifest_streams = {
        (str(row["topic_id"]), str(row["variant"]), RETRIEVER_NAME): str(row["family"])
        for row in inputs.manifest["streams"]  # type: ignore[index]
    }
    if len(manifest_streams) != 31:
        raise ValueError("manifest stream identities are not exactly 31 unique streams")
    candidates_by_stream: dict[
        tuple[str, str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    for candidate in inputs.candidates:
        identity = (
            str(candidate["topic_id"]),
            str(candidate["variant"]),
            RETRIEVER_NAME,
        )
        if identity not in manifest_streams:
            raise ValueError("candidate row is outside the manifest stream identities")
        if str(candidate["family"]) != manifest_streams[identity]:
            raise ValueError("candidate family differs from the manifest")
        candidates_by_stream[identity].append(candidate)
    if set(candidates_by_stream) != set(manifest_streams):
        raise ValueError("candidate snapshot does not cover every manifest stream")

    score_by_window = {_window_identity(row): row for row in inputs.scores}
    windows_by_stream_doc: dict[
        tuple[str, str, str, str], list[dict[str, object]]
    ] = defaultdict(list)
    for window in inputs.windows:
        score_row = score_by_window[_window_identity(window)]
        score = float(score_row["score"])
        if not math.isfinite(score):
            raise ValueError("MiniLM scores must be finite raw logits")
        if (
            str(window["family"]) != str(score_row["family"])
            or int(window["rank"]) != int(score_row["rank"])
        ):
            raise ValueError("score row lineage differs from its preflight window")
        merged = dict(window)
        merged["score"] = score
        windows_by_stream_doc[
            (
                str(window["topic_id"]),
                str(window["variant"]),
                RETRIEVER_NAME,
                str(window["document_id"]),
            )
        ].append(merged)

    result: dict[tuple[str, str, str], dict[str, object]] = {}
    for identity in sorted(manifest_streams):
        candidates = sorted(
            candidates_by_stream[identity],
            key=lambda row: (int(row["rank"]), str(row["document_id"])),
        )
        if (
            len(candidates) != 100
            or [int(row["rank"]) for row in candidates] != list(range(1, 101))
            or len({str(row["document_id"]) for row in candidates}) != 100
        ):
            raise ValueError(f"stream {identity!r} is not exact depth 100")
        stream_windows: list[Mapping[str, object]] = []
        for candidate in candidates:
            key = (*identity, str(candidate["document_id"]))
            document_windows = windows_by_stream_doc.get(key, [])
            if not document_windows:
                raise ValueError("candidate lacks scored preflight windows")
            if any(
                str(window["query_sha256"]) != str(candidate["query_sha256"])
                or int(window["rank"]) != int(candidate["rank"])
                for window in document_windows
            ):
                raise ValueError("candidate/window query or prior-rank lineage differs")
            stream_windows.extend(document_windows)

        bm25: list[dict[str, object]] = []
        for candidate in candidates:
            first_window = min(
                windows_by_stream_doc[
                    (*identity, str(candidate["document_id"]))
                ],
                key=lambda row: (
                    int(row["document_start_token"]),
                    str(row["window_id"]),
                ),
            )
            row = dict(candidate)
            row.update(
                {
                    "aggregation": "bm25",
                    "passage": str(first_window["window_text"]),
                    "prior_rank": int(candidate["rank"]),
                    "score": float(candidate["source_score"]),
                    "selected_windows": [],
                }
            )
            bm25.append(row)
        top4 = rerank_stream(candidates, stream_windows, aggregation="top4")
        maxp = rerank_stream(candidates, stream_windows, aggregation="maxp")
        result[identity] = {
            "bm25": tuple(bm25),
            "family": manifest_streams[identity],
            "maxp": tuple(maxp),
            "top4": tuple(top4),
        }
    return result


_LEGACY_RANKING_FILE_SHA256 = (
    "1edd1542777ae45c819905b697fcec0242de5421b34e7531fbf0178395837bed"
)
_LEGACY_FREEZE_FILE_SHA256 = (
    "4a78b44ede4b979a3cb3ec96348088e4e08626e2cc4c92b464c6e36097a71389"
)
_LEGACY_CANONICAL_RANKING_SHA256 = (
    "6714c18f7c78542ccf023f2c0a7f0a727fb84af168753a0779a49a0f51137153"
)


def _canonical_json_bytes(value: object, *, pretty: bool = True) -> bytes:
    options: dict[str, object] = {
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _artifact_record(content: bytes, *, rows: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _write_artifact(
    root: Path,
    relative: str,
    content: bytes,
    artifacts: dict[str, dict[str, object]],
    *,
    rows: int | None = None,
) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    record = _artifact_record(content, rows=rows)
    artifacts[relative] = record
    return record


def _weight_payloads(
    legacy: Mapping[tuple[str, str], Fraction],
    corrected: Mapping[tuple[str, str, str], Fraction],
    manifest_streams: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    topics = sorted({str(stream["topic_id"]) for stream in manifest_streams})
    legacy_rows = [
        {
            "denominator": weight.denominator,
            "numerator": weight.numerator,
            "retriever_name": retriever,
            "variant_name": variant,
            "weight": float(weight),
        }
        for (variant, retriever), weight in sorted(legacy.items())
    ]
    effective_checks = {}
    for topic_id in topics:
        active = [
            (str(stream["variant"]), RETRIEVER_NAME)
            for stream in manifest_streams
            if str(stream["topic_id"]) == topic_id
        ]
        total = sum((legacy[identity] for identity in active), Fraction(0, 1))
        effective_checks[topic_id] = {
            "denominator": total.denominator,
            "numerator": total.numerator,
            "sum": float(total),
        }
    legacy_payload = {
        "effective_topic_sum_checks": effective_checks,
        "facet_family_weight": 0.5,
        "implementation_identity": ["variant_name", "retriever_name"],
        "original_family_weight": 0.5,
        "schema_version": "family-rrf-global-key-v1",
        "weights": legacy_rows,
    }

    corrected_rows = [
        {
            "denominator": weight.denominator,
            "numerator": weight.numerator,
            "retriever_name": retriever,
            "topic_id": topic_id,
            "variant_name": variant,
            "weight": float(weight),
        }
        for (topic_id, variant, retriever), weight in sorted(corrected.items())
    ]
    topic_checks = {}
    for topic_id in topics:
        topic_weights = [
            weight
            for (topic, _variant, _retriever), weight in corrected.items()
            if topic == topic_id
        ]
        total = sum(topic_weights, Fraction(0, 1))
        originals = [
            weight
            for (topic, variant, _retriever), weight in corrected.items()
            if topic == topic_id
            and any(
                str(stream["topic_id"]) == topic_id
                and str(stream["variant"]) == variant
                and str(stream["family"]) == "original"
                for stream in manifest_streams
            )
        ]
        topic_checks[topic_id] = {
            "denominator": total.denominator,
            "exact_sum": f"{total.numerator}/{total.denominator}",
            "numerator": total.numerator,
            "original_weight": float(sum(originals, Fraction(0, 1))),
            "sum": float(total),
        }
        if total != Fraction(1, 1) or sum(originals, Fraction(0, 1)) != Fraction(1, 2):
            raise ValueError("corrected topic-local weights do not sum exactly to 1.0")
    corrected_payload = {
        "facet_family_weight": 0.5,
        "implementation_identity": [
            "topic_id",
            "variant_name",
            "retriever_name",
        ],
        "original_family_weight": 0.5,
        "schema_version": "family-rrf-topic-local-v2",
        "topic_sum_checks": topic_checks,
        "weights": corrected_rows,
    }
    return legacy_payload, corrected_payload


def _weight_map(payload: Mapping[str, object]) -> dict[tuple[str, ...], Fraction]:
    result: dict[tuple[str, ...], Fraction] = {}
    for raw in payload["weights"]:  # type: ignore[index]
        if not isinstance(raw, Mapping):
            raise ValueError("fusion weight row must be an object")
        if "topic_id" in raw:
            key = (
                str(raw["topic_id"]),
                str(raw["variant_name"]),
                str(raw["retriever_name"]),
            )
        else:
            key = (str(raw["variant_name"]), str(raw["retriever_name"]))
        value = Fraction(int(raw["numerator"]), int(raw["denominator"]))
        if key in result or float(value) != float(raw["weight"]):
            raise ValueError("fusion weight table is duplicate or internally inconsistent")
        result[key] = value
    return result


def _project_stream_row(row: Mapping[str, object]) -> dict[str, object]:
    projected = dict(row)
    projected.pop("text", None)
    return projected


def _safe_artifact_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("freeze artifact path must be a non-empty relative path")
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("freeze artifact path escapes the freeze directory") from exc
    return candidate


def _legacy_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    freeze_root = repo_root / "outputs/rag25_sparse_relevance_paired_v1/freeze_v1"
    return freeze_root / "freeze.json", freeze_root / "rankings/R1__family_rrf.jsonl"


def generate_freeze(
    *,
    manifest_path: Path,
    preflight_dir: Path,
    scores_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create the complete pre-qrels ranking freeze without new external work."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"create-only freeze already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent)
    )
    try:
        inputs = load_authenticated_inputs(
            manifest_path=manifest_path,
            preflight_dir=preflight_dir,
            scores_dir=scores_dir,
        )
        streams = build_stream_rankings(inputs)
        legacy_weights, corrected_weights = build_weight_tables(
            inputs.manifest["streams"],  # type: ignore[index]
            retriever_name=RETRIEVER_NAME,
        )
        legacy_weight_payload, corrected_weight_payload = _weight_payloads(
            legacy_weights,
            corrected_weights,
            inputs.manifest["streams"],  # type: ignore[index]
        )

        legacy_freeze_path, legacy_ranking_path = _legacy_paths()
        if (
            _sha256_file(legacy_freeze_path) != _LEGACY_FREEZE_FILE_SHA256
            or inputs.manifest.get("prior_freeze_sha256")
            != _LEGACY_FREEZE_FILE_SHA256
            or _sha256_file(legacy_ranking_path) != _LEGACY_RANKING_FILE_SHA256
        ):
            raise ValueError("historical R1 artifacts differ from the authenticated baseline")
        legacy_freeze = _read_json(legacy_freeze_path, "historical R1 freeze")
        if (
            legacy_freeze.get("rankings", {})  # type: ignore[union-attr]
            .get("R1:family_rrf", {})
            .get("sha256")
            != _LEGACY_CANONICAL_RANKING_SHA256
        ):
            raise ValueError("historical R1 canonical hash differs")
        reconstructed_legacy = reconstruct_legacy_ranking(
            inputs.candidates, legacy_weights
        )
        if canonical_ranking_sha256(reconstructed_legacy) != _LEGACY_CANONICAL_RANKING_SHA256:
            raise ValueError("legacy v1 weights do not independently reconstruct R1")
        legacy_bytes = legacy_ranking_path.read_bytes()
        historical_rows = tuple(
            json.loads(line) for line in legacy_bytes.splitlines() if line
        )
        if [
            (row["topic_id"], row["rank"], row["docid"])
            for row in reconstructed_legacy
        ] != [
            (row["topic_id"], row["rank"], row["docid"])
            for row in historical_rows
        ]:
            raise ValueError("legacy R1 canonical projection differs on reconstruction")

        matrix = build_arm_streams(streams)
        rankings: dict[str, Sequence[Mapping[str, object]]] = {
            "R1_LEGACY": historical_rows
        }
        for arm in ARM_NAMES:
            if arm == "R1_LEGACY":
                continue
            arm_weights: Mapping[tuple[str, ...], Fraction]
            if arm == "BF100_LEGACY_FUSION":
                arm_weights = legacy_weights
            else:
                arm_weights = corrected_weights
            rankings[arm] = fuse_streams(matrix[arm], arm_weights)
        for arm, rows in rankings.items():
            if (
                len(rows) != 400
                or {str(row["topic_id"]) for row in rows} != set(PILOT_TOPIC_IDS)
                or any(
                    [int(row["rank"]) for row in rows if str(row["topic_id"]) == topic]
                    != list(range(1, 101))
                    for topic in PILOT_TOPIC_IDS
                )
            ):
                raise ValueError(f"ranking {arm} does not have exact four-topic depth 100")

        control_diff = build_control_diff(
            rankings["R1_LEGACY"], rankings["C0_TOPIC_LOCAL"]
        )
        fusion_definitions = {
            "deduplication": "document_id within topic after one best rank per stream",
            "facet_family_weight": 0.5,
            "k": 60,
            "output_depth": 100,
            "primary_implementation": "family_rrf_topic_local_v2",
            "raw_score_cross_stream_comparison": False,
            "secondary_implementation": "family_rrf_global_key_v1",
            "stream_rank_tie_break": [
                "descending_aggregate_score",
                "prior_stream_rank",
                "document_id",
            ],
            "system_rank_tie_break": [
                "descending_rrf_score",
                "best_stream_rank",
                "document_id",
            ],
        }

        artifacts: dict[str, dict[str, object]] = {}
        _write_artifact(
            stage,
            "input_bindings.json",
            _canonical_json_bytes(dict(inputs.bindings)),
            artifacts,
        )
        _write_artifact(
            stage,
            "fusion_definitions.json",
            _canonical_json_bytes(fusion_definitions),
            artifacts,
        )
        _write_artifact(
            stage,
            "fusion_weights_v1.json",
            _canonical_json_bytes(legacy_weight_payload),
            artifacts,
        )
        _write_artifact(
            stage,
            "fusion_weights_v2.json",
            _canonical_json_bytes(corrected_weight_payload),
            artifacts,
        )
        _write_artifact(
            stage,
            "legacy_corrected_diff.json",
            _canonical_json_bytes(control_diff),
            artifacts,
        )

        stream_records: list[dict[str, object]] = []
        for identity in sorted(streams):
            topic_id, variant, retriever = identity
            variant_digest = hashlib.sha256(
                f"{variant}\0{retriever}".encode("utf-8")
            ).hexdigest()[:16]
            for aggregation in ("bm25", "top4", "maxp"):
                rows = [
                    _project_stream_row(row)
                    for row in streams[identity][aggregation]  # type: ignore[index]
                ]
                relative = (
                    f"streams/{aggregation}/{topic_id}__{variant_digest}.jsonl"
                )
                content = _jsonl_bytes(rows)
                record = _write_artifact(
                    stage, relative, content, artifacts, rows=len(rows)
                )
                stream_records.append(
                    {
                        "aggregation": aggregation,
                        "family": streams[identity]["family"],
                        "file_sha256": record["sha256"],
                        "path": relative,
                        "retriever_name": retriever,
                        "rows": len(rows),
                        "topic_id": topic_id,
                        "variant_name": variant,
                    }
                )
        _write_artifact(
            stage,
            "stream_manifest.json",
            _canonical_json_bytes(
                {
                    "schema_version": "facet-local-minilm-stream-manifest-v1",
                    "streams": stream_records,
                }
            ),
            artifacts,
        )

        ranking_records: dict[str, dict[str, object]] = {}
        for arm in ARM_NAMES:
            relative = f"rankings/{arm}.jsonl"
            rows = rankings[arm]
            content = legacy_bytes if arm == "R1_LEGACY" else _jsonl_bytes(rows)
            record = _write_artifact(
                stage, relative, content, artifacts, rows=len(rows)
            )
            ranking_records[arm] = {
                "canonical_sha256": canonical_ranking_sha256(rows),
                "file_sha256": record["sha256"],
                "path": relative,
                "rows": len(rows),
            }

        payload: dict[str, object] = {
            "artifacts": artifacts,
            "bindings": dict(inputs.bindings),
            "control_diff_audit": "legacy_corrected_diff.json",
            "fusion_definitions": "fusion_definitions.json",
            "fusion_tables": {
                "family_rrf_global_key_v1": "fusion_weights_v1.json",
                "family_rrf_topic_local_v2": "fusion_weights_v2.json",
            },
            "qrels_opened": False,
            "rankings": ranking_records,
            "schema_version": "facet-local-minilm-ranking-freeze-v1",
            "status": "frozen_before_qrels",
            "streams": stream_records,
            "topic_ids": list(PILOT_TOPIC_IDS),
        }
        payload["freeze_sha256"] = hashlib.sha256(
            _canonical_json_bytes(payload, pretty=False)
        ).hexdigest()
        (stage / "freeze.json").write_bytes(_canonical_json_bytes(payload))
        if output_dir.exists():
            raise FileExistsError(f"create-only freeze already exists: {output_dir}")
        os.rename(stage, output_dir)
        return payload
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def verify_freeze(freeze_dir: Path) -> dict[str, object]:
    """Independently replay and verify a self-contained Task 4 ranking freeze."""

    freeze_dir = Path(freeze_dir)
    freeze_path = freeze_dir / "freeze.json"
    payload = _read_json(freeze_path, "ranking freeze")
    if freeze_path.read_bytes() != _canonical_json_bytes(payload):
        raise ValueError("ranking freeze root is not canonical JSON")
    frozen_hash = payload.get("freeze_sha256")
    without_hash = dict(payload)
    without_hash.pop("freeze_sha256", None)
    if frozen_hash != hashlib.sha256(
        _canonical_json_bytes(without_hash, pretty=False)
    ).hexdigest():
        raise ValueError("ranking freeze self hash differs")
    if (
        payload.get("schema_version") != "facet-local-minilm-ranking-freeze-v1"
        or payload.get("status") != "frozen_before_qrels"
        or payload.get("qrels_opened") is not False
        or tuple(payload.get("topic_ids", ())) != PILOT_TOPIC_IDS
        or set(payload.get("rankings", {})) != set(ARM_NAMES)
        or len(payload.get("streams", ())) != 93
    ):
        raise ValueError("ranking freeze root differs from the exact Task 4 contract")
    serialized_root = json.dumps(payload, sort_keys=True)
    if "qrels_path" in serialized_root:
        raise ValueError("ranking freeze must not contain a qrels path")

    artifact_records = payload.get("artifacts")
    if not isinstance(artifact_records, Mapping):
        raise ValueError("ranking freeze lacks artifact hash bindings")
    for relative, raw_record in artifact_records.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError("artifact record must be an object")
        path = _safe_artifact_path(freeze_dir, relative)
        content = path.read_bytes()
        if (
            len(content) != raw_record.get("bytes")
            or hashlib.sha256(content).hexdigest() != raw_record.get("sha256")
        ):
            raise ValueError(f"artifact hash differs for {relative}")

    stream_records = payload["streams"]
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for raw in stream_records:  # type: ignore[union-attr]
        if not isinstance(raw, Mapping):
            raise ValueError("stream manifest record must be an object")
        identity = (
            str(raw["topic_id"]),
            str(raw["variant_name"]),
            str(raw["retriever_name"]),
        )
        aggregation = str(raw["aggregation"])
        if aggregation not in {"bm25", "top4", "maxp"} or raw.get("rows") != 100:
            raise ValueError("stream artifact differs from exact depth 100")
        content = _safe_artifact_path(freeze_dir, raw["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != raw.get("file_sha256"):
            raise ValueError("stream manifest hash differs from artifact")
        rows = tuple(json.loads(line) for line in content.splitlines() if line)
        if len(rows) != 100:
            raise ValueError("stream artifact row count differs")
        record = grouped.setdefault(identity, {"family": str(raw["family"])})
        if aggregation in record:
            raise ValueError("duplicate stream aggregation artifact")
        record[aggregation] = rows
    if len(grouped) != 31 or any(
        set(record) != {"family", "bm25", "top4", "maxp"}
        for record in grouped.values()
    ):
        raise ValueError("stream artifacts do not form exactly 31 complete streams")

    fusion_tables = payload.get("fusion_tables")
    if fusion_tables != {
        "family_rrf_global_key_v1": "fusion_weights_v1.json",
        "family_rrf_topic_local_v2": "fusion_weights_v2.json",
    }:
        raise ValueError("fusion table identities differ")
    legacy_payload = _read_json(
        _safe_artifact_path(
            freeze_dir,
            fusion_tables["family_rrf_global_key_v1"],  # type: ignore[index]
        ),
        "legacy weights",
    )
    corrected_payload = _read_json(
        _safe_artifact_path(
            freeze_dir,
            fusion_tables["family_rrf_topic_local_v2"],  # type: ignore[index]
        ),
        "topic-local weights",
    )
    if (
        legacy_payload.get("schema_version") != "family-rrf-global-key-v1"
        or corrected_payload.get("schema_version") != "family-rrf-topic-local-v2"
    ):
        raise ValueError("fusion table schema differs")
    legacy_weights = _weight_map(legacy_payload)
    corrected_weights = _weight_map(corrected_payload)
    for topic_id in PILOT_TOPIC_IDS:
        topic_sum = sum(
            (
                value
                for (topic, _variant, _retriever), value in corrected_weights.items()
                if topic == topic_id
            ),
            Fraction(0, 1),
        )
        original_sum = sum(
            (
                corrected_weights[identity]
                for identity, record in grouped.items()
                if identity[0] == topic_id and record["family"] == "original"
            ),
            Fraction(0, 1),
        )
        if topic_sum != Fraction(1, 1) or original_sum != Fraction(1, 2):
            raise ValueError("topic-local weights fail exact sum checks")

    matrix = build_arm_streams(grouped)
    ranking_records = payload["rankings"]
    decoded_rankings: dict[str, tuple[Mapping[str, object], ...]] = {}
    for arm in ARM_NAMES:
        raw_record = ranking_records[arm]  # type: ignore[index]
        path = _safe_artifact_path(freeze_dir, raw_record["path"])
        content = path.read_bytes()
        if (
            len(content.splitlines()) != 400
            or hashlib.sha256(content).hexdigest() != raw_record["file_sha256"]
            or raw_record["rows"] != 400
        ):
            raise ValueError(f"ranking artifact differs for {arm}")
        rows = tuple(json.loads(line) for line in content.splitlines() if line)
        if canonical_ranking_sha256(rows) != raw_record["canonical_sha256"]:
            raise ValueError(f"ranking canonical hash differs for {arm}")
        decoded_rankings[arm] = rows
        if arm == "R1_LEGACY":
            if raw_record["file_sha256"] != _LEGACY_RANKING_FILE_SHA256:
                raise ValueError("R1_LEGACY is not byte-identical to historical R1")
            continue
        arm_weights = (
            legacy_weights if arm == "BF100_LEGACY_FUSION" else corrected_weights
        )
        reconstructed = fuse_streams(matrix[arm], arm_weights)
        if _jsonl_bytes(reconstructed) != content:
            raise ValueError(f"ranking {arm} does not replay from frozen streams")

    reconstructed_legacy_projection = fuse_streams(
        matrix["C0_TOPIC_LOCAL"], legacy_weights
    )
    if [
        (row["topic_id"], row["rank"], row["docid"])
        for row in reconstructed_legacy_projection
    ] != [
        (row["topic_id"], row["rank"], row["docid"])
        for row in decoded_rankings["R1_LEGACY"]
    ]:
        raise ValueError("legacy v1 table does not reproduce historical R1 projection")

    expected_audit = build_control_diff(
        decoded_rankings["R1_LEGACY"], decoded_rankings["C0_TOPIC_LOCAL"]
    )
    audit_path = _safe_artifact_path(freeze_dir, payload["control_diff_audit"])
    if audit_path.read_bytes() != _canonical_json_bytes(expected_audit):
        raise ValueError("legacy/corrected qrels-free diff audit differs")
    return {
        "freeze_sha256": frozen_hash,
        "qrels_opened": False,
        "ranking_count": len(decoded_rankings),
        "status": "verified",
        "stream_ranking_count": len(stream_records),  # type: ignore[arg-type]
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run create-only ranking materialization or independent freeze verification."""

    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "verify":
        parser = argparse.ArgumentParser(description="Verify a facet-local MiniLM freeze")
        parser.add_argument("verify", nargs="?")
        parser.add_argument("--freeze", type=Path, required=True)
        args = parser.parse_args(arguments)
        print(json.dumps(verify_freeze(args.freeze), sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(description="Freeze facet-local MiniLM rankings")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    payload = generate_freeze(
        manifest_path=args.manifest,
        preflight_dir=args.preflight,
        scores_dir=args.scores,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "freeze_sha256": payload["freeze_sha256"],
                "output": str(args.output),
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
