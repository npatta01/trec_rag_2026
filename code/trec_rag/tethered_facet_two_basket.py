"""Deterministic protected two-basket rankings for tethered facet scores."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = "tethered-facet-two-basket-freeze-v1"
SEAL_SCHEMA_VERSION = "tethered-facet-two-basket-seal-v1"
ARMS = ("FACET-2B", "TETHERED-2B")
PILOT_TOPIC_IDS = ("219", "72", "300", "84")
HEAD_SIZE = 100
BASKET_SIZE = 200
INTERLEAVED_END = HEAD_SIZE + (2 * BASKET_SIZE)
PREFIX_DEPTHS = (100, 500, 1000)
TASK2_SCORE_SCHEMA_VERSION = "tethered-facet-minilm-document-score-v1"


@dataclass(frozen=True)
class FacetScores:
    """One query's raw document scores and authenticated lineage."""

    facet_id: str
    manifest_order: int
    scores: Mapping[str, float]
    bm25_ranks: Mapping[str, int]
    query_sha256: str
    text_sha256: Mapping[str, str]
    model: str
    model_revision: str
    score_schema_version: str


@dataclass(frozen=True)
class TopicInput:
    """Previously frozen populations plus one arm's facet score streams."""

    topic_id: str
    accepted_union: tuple[str, ...]
    rrf: tuple[str, ...]
    dual: tuple[str, ...]
    facets: tuple[FacetScores, ...]


@dataclass(frozen=True)
class BasketSelection:
    document_id: str
    facet_id: str
    percentile: float
    facet_rank: int
    prior_bm25_rank: int
    nominal_quota: int
    shortage: int
    duplicate_skip_delta: int
    query_sha256: str
    text_sha256: str
    model: str
    model_revision: str
    score_schema_version: str


@dataclass(frozen=True)
class FacetBasket:
    selections: tuple[BasketSelection, ...]
    duplicate_skip_totals: tuple[tuple[str, int], ...]

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(row.document_id for row in self.selections)


@dataclass(frozen=True)
class RankingEntry:
    document_id: str
    source: str
    rrf_rank: int
    dual_rank: int
    facet: BasketSelection | None = None


@dataclass(frozen=True)
class TwoBasketPermutation:
    entries: tuple[RankingEntry, ...]
    duplicate_skip_totals: tuple[tuple[str, int], ...]

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(row.document_id for row in self.entries)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(row.source for row in self.entries)


def average_rank_percentiles(
    scores: Mapping[str, float | int],
) -> dict[str, float]:
    """Return within-query average-rank percentiles, with larger scores best."""

    if not scores:
        return {}
    groups: dict[float, list[str]] = defaultdict(list)
    for document_id, raw_score in scores.items():
        score = float(raw_score)
        if not document_id or not math.isfinite(score):
            raise ValueError("scores require non-empty document IDs and finite values")
        groups[score].append(document_id)
    count = len(scores)
    first_rank = 1
    output: dict[str, float] = {}
    for score in sorted(groups, reverse=True):
        last_rank = first_rank + len(groups[score]) - 1
        percentile = (count - ((first_rank + last_rank) / 2) + 1) / count
        for document_id in groups[score]:
            output[document_id] = percentile
        first_rank = last_rank + 1
    return output


def topic_quotas(facet_count: int, capacity: int = 200) -> list[int]:
    """Divide capacity in manifest order, assigning remainder slots first."""

    if facet_count <= 0 or capacity <= 0:
        raise ValueError("positive facet count and capacity are required")
    base, remainder = divmod(capacity, facet_count)
    return [base + (index < remainder) for index in range(facet_count)]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())


def _validate_topic(topic: TopicInput) -> None:
    if topic.topic_id not in PILOT_TOPIC_IDS:
        raise ValueError(f"topic {topic.topic_id} is outside the protected pilot")
    population = set(topic.accepted_union)
    if (
        len(population) != len(topic.accepted_union)
        or not population
        or len(topic.rrf) != len(population)
        or len(set(topic.rrf)) != len(population)
        or set(topic.rrf) != population
        or len(topic.dual) != len(population)
        or len(set(topic.dual)) != len(population)
        or set(topic.dual) != population
    ):
        raise ValueError("RRF and DUAL must be complete permutations of accepted union")
    if len(population) < INTERLEAVED_END:
        raise ValueError("accepted union must contain at least 500 documents")
    facet_ids: set[str] = set()
    facet_orders: set[int] = set()
    for facet in topic.facets:
        if (
            not facet.facet_id
            or facet.facet_id in facet_ids
            or facet.manifest_order < 0
            or facet.manifest_order in facet_orders
        ):
            raise ValueError("facets require unique identities and manifest orders")
        facet_ids.add(facet.facet_id)
        facet_orders.add(facet.manifest_order)
        score_ids = set(facet.scores)
        if (
            not score_ids
            or score_ids != set(facet.bm25_ranks)
            or score_ids != set(facet.text_sha256)
            or not score_ids <= population
            or not _valid_sha256(facet.query_sha256)
            or not facet.model
            or not facet.model_revision
            or not facet.score_schema_version
        ):
            raise ValueError("facet score population or provenance is invalid")
        ranks = list(facet.bm25_ranks.values())
        if (
            any(type(rank) is not int or rank <= 0 for rank in ranks)
            or len(set(ranks)) != len(ranks)
            or any(not _valid_sha256(value) for value in facet.text_sha256.values())
        ):
            raise ValueError("facet rank or text provenance is invalid")


def _facet_edges(
    topic: TopicInput,
) -> tuple[list[tuple[float, int, int, str, FacetScores, int]], list[FacetScores]]:
    facets = sorted(topic.facets, key=lambda value: (value.manifest_order, value.facet_id))
    edges: list[tuple[float, int, int, str, FacetScores, int]] = []
    for facet in facets:
        percentiles = average_rank_percentiles(facet.scores)
        facet_order = sorted(
            facet.scores,
            key=lambda document_id: (
                -float(facet.scores[document_id]),
                facet.bm25_ranks[document_id],
                document_id,
            ),
        )
        ranks = {document_id: rank for rank, document_id in enumerate(facet_order, 1)}
        for document_id in facet.scores:
            edges.append(
                (
                    -percentiles[document_id],
                    facet.manifest_order,
                    facet.bm25_ranks[document_id],
                    document_id,
                    facet,
                    ranks[document_id],
                )
            )
    edges.sort(key=lambda edge: edge[:4])
    return edges, facets


def build_facet_basket(topic: TopicInput) -> FacetBasket:
    """Select a quota-balanced facet basket without crossing raw-score queries."""

    _validate_topic(topic)
    edges, facets = _facet_edges(topic)
    if not facets:
        raise ValueError("two-basket ranking requires at least one accepted facet")
    quotas = topic_quotas(len(facets), BASKET_SIZE)
    quota_by_id = {facet.facet_id: quotas[index] for index, facet in enumerate(facets)}
    excluded = set(topic.rrf[: HEAD_SIZE + BASKET_SIZE])
    selected_ids: set[str] = set()
    selected_edges: list[
        tuple[tuple[float, int, int, str, FacetScores, int], int]
    ] = []
    counts = {facet.facet_id: 0 for facet in facets}
    duplicate_skips = {facet.facet_id: 0 for facet in facets}
    pending_duplicate_skips = {facet.facet_id: 0 for facet in facets}
    processed = {facet.facet_id: set() for facet in facets}

    for edge in edges:
        document_id, facet = edge[3], edge[4]
        if document_id in excluded:
            processed[facet.facet_id].add(document_id)
            continue
        if counts[facet.facet_id] >= quota_by_id[facet.facet_id]:
            continue
        processed[facet.facet_id].add(document_id)
        if document_id in selected_ids:
            duplicate_skips[facet.facet_id] += 1
            pending_duplicate_skips[facet.facet_id] += 1
            continue
        selected_edges.append((edge, pending_duplicate_skips[facet.facet_id]))
        pending_duplicate_skips[facet.facet_id] = 0
        selected_ids.add(document_id)
        counts[facet.facet_id] += 1

    shortages = {
        facet.facet_id: quota_by_id[facet.facet_id] - counts[facet.facet_id]
        for facet in facets
    }
    # Any nominal shortage becomes a shared pool.  Each manifest-order pass
    # gives at most one extra slot to a facet, preventing one stream from
    # consuming the entire redistribution before the next stream is visited.
    by_facet = {
        facet.facet_id: [edge for edge in edges if edge[4].facet_id == facet.facet_id]
        for facet in facets
    }
    cursors = {facet.facet_id: 0 for facet in facets}
    while len(selected_edges) < BASKET_SIZE:
        progress = False
        for facet in facets:
            if len(selected_edges) == BASKET_SIZE:
                break
            facet_edges = by_facet[facet.facet_id]
            while cursors[facet.facet_id] < len(facet_edges):
                edge = facet_edges[cursors[facet.facet_id]]
                cursors[facet.facet_id] += 1
                document_id = edge[3]
                if document_id in processed[facet.facet_id]:
                    continue
                processed[facet.facet_id].add(document_id)
                if document_id in excluded:
                    continue
                if document_id in selected_ids:
                    duplicate_skips[facet.facet_id] += 1
                    pending_duplicate_skips[facet.facet_id] += 1
                    continue
                selected_edges.append(
                    (edge, pending_duplicate_skips[facet.facet_id])
                )
                pending_duplicate_skips[facet.facet_id] = 0
                selected_ids.add(document_id)
                counts[facet.facet_id] += 1
                progress = True
                break
        if not progress:
            raise ValueError("facet streams cannot fill the exact 200-document basket")

    selections = tuple(
        BasketSelection(
            document_id=edge[3],
            facet_id=edge[4].facet_id,
            percentile=-edge[0],
            facet_rank=edge[5],
            prior_bm25_rank=edge[2],
            nominal_quota=quota_by_id[edge[4].facet_id],
            shortage=shortages[edge[4].facet_id],
            duplicate_skip_delta=duplicate_skip_delta,
            query_sha256=edge[4].query_sha256,
            text_sha256=edge[4].text_sha256[edge[3]],
            model=edge[4].model,
            model_revision=edge[4].model_revision,
            score_schema_version=edge[4].score_schema_version,
        )
        for edge, duplicate_skip_delta in selected_edges
    )
    return FacetBasket(
        selections,
        tuple(
            (facet.facet_id, duplicate_skips[facet.facet_id]) for facet in facets
        ),
    )


def build_two_basket_permutation(topic: TopicInput) -> TwoBasketPermutation:
    """Protect RRF ranks 1--100, alternate two baskets, then use DUAL order."""

    _validate_topic(topic)
    facet_basket = build_facet_basket(topic)
    rrf_rank = {document_id: rank for rank, document_id in enumerate(topic.rrf, 1)}
    dual_rank = {document_id: rank for rank, document_id in enumerate(topic.dual, 1)}
    entries = [
        RankingEntry(document_id, "rrf_head", rrf_rank[document_id], dual_rank[document_id])
        for document_id in topic.rrf[:HEAD_SIZE]
    ]
    for rrf_document, facet in zip(
        topic.rrf[HEAD_SIZE : HEAD_SIZE + BASKET_SIZE],
        facet_basket.selections,
        strict=True,
    ):
        entries.append(
            RankingEntry(
                rrf_document,
                "rrf_basket",
                rrf_rank[rrf_document],
                dual_rank[rrf_document],
            )
        )
        entries.append(
            RankingEntry(
                facet.document_id,
                "facet_basket",
                rrf_rank[facet.document_id],
                dual_rank[facet.document_id],
                facet,
            )
        )
    selected = {entry.document_id for entry in entries}
    entries.extend(
        RankingEntry(document_id, "dual_tail", rrf_rank[document_id], dual_rank[document_id])
        for document_id in topic.dual
        if document_id not in selected
    )
    result = TwoBasketPermutation(tuple(entries), facet_basket.duplicate_skip_totals)
    if (
        len(result.document_ids) != len(set(result.document_ids))
        or set(result.document_ids) != set(topic.accepted_union)
    ):
        raise ValueError("two-basket result is not a complete duplicate-free permutation")
    return result


def _entry_row(topic_id: str, arm: str, rank: int, entry: RankingEntry) -> dict[str, object]:
    facet = entry.facet
    return {
        "schema_version": SCHEMA_VERSION,
        "topic_id": topic_id,
        "arm": arm,
        "rank": rank,
        "document_id": entry.document_id,
        "source": entry.source,
        "generating_facet": facet.facet_id if facet else None,
        "percentile": facet.percentile if facet else None,
        "facet_rank": facet.facet_rank if facet else None,
        "prior_bm25_rank": facet.prior_bm25_rank if facet else None,
        "nominal_quota": facet.nominal_quota if facet else None,
        "shortage": facet.shortage if facet else None,
        "duplicate_skip_delta": facet.duplicate_skip_delta if facet else None,
        "rrf_rank": entry.rrf_rank,
        "dual_rank": entry.dual_rank,
        "query_sha256": facet.query_sha256 if facet else None,
        "text_sha256": facet.text_sha256 if facet else None,
        "score_provenance": (
            {
                "schema_version": facet.score_schema_version,
                "model": facet.model,
                "model_revision": facet.model_revision,
                "score_space": "raw_query_local_logits",
            }
            if facet
            else None
        ),
    }


def _binding(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    content = resolved.read_bytes()
    return {"path": str(resolved), "bytes": len(content), "sha256": _sha256(content)}


def _topic_payload(topic: TopicInput) -> dict[str, object]:
    return {
        "topic_id": topic.topic_id,
        "accepted_union": list(topic.rrf),
        "rrf": list(topic.rrf),
        "dual": list(topic.dual),
        "facets": [
            {
                "facet_id": facet.facet_id,
                "manifest_order": facet.manifest_order,
                "scores": dict(facet.scores),
                "bm25_ranks": dict(facet.bm25_ranks),
                "query_sha256": facet.query_sha256,
                "text_sha256": dict(facet.text_sha256),
                "model": facet.model,
                "model_revision": facet.model_revision,
                "score_schema_version": facet.score_schema_version,
            }
            for facet in sorted(
                topic.facets, key=lambda value: (value.manifest_order, value.facet_id)
            )
        ],
    }


def _topic_from_payload(value: object) -> TopicInput:
    if not isinstance(value, Mapping) or not isinstance(value.get("facets"), list):
        raise ValueError("frozen semantic topic input is invalid")
    facets: list[FacetScores] = []
    for raw in value["facets"]:
        if not isinstance(raw, Mapping):
            raise ValueError("frozen semantic facet input is invalid")
        try:
            facets.append(
                FacetScores(
                    facet_id=str(raw["facet_id"]),
                    manifest_order=int(raw["manifest_order"]),
                    scores={str(key): float(score) for key, score in raw["scores"].items()},  # type: ignore[union-attr]
                    bm25_ranks={str(key): int(rank) for key, rank in raw["bm25_ranks"].items()},  # type: ignore[union-attr]
                    query_sha256=str(raw["query_sha256"]),
                    text_sha256={str(key): str(digest) for key, digest in raw["text_sha256"].items()},  # type: ignore[union-attr]
                    model=str(raw["model"]),
                    model_revision=str(raw["model_revision"]),
                    score_schema_version=str(raw["score_schema_version"]),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("frozen semantic facet input is invalid") from exc
    try:
        topic = TopicInput(
            topic_id=str(value["topic_id"]),
            accepted_union=tuple(str(item) for item in value["accepted_union"]),  # type: ignore[union-attr]
            rrf=tuple(str(item) for item in value["rrf"]),  # type: ignore[union-attr]
            dual=tuple(str(item) for item in value["dual"]),  # type: ignore[union-attr]
            facets=tuple(facets),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("frozen semantic topic input is invalid") from exc
    _validate_topic(topic)
    return topic


def _validate_arm_contract(by_arm: Mapping[str, Mapping[str, TopicInput]]) -> None:
    for topic_id in PILOT_TOPIC_IDS:
        baseline = by_arm["FACET-2B"][topic_id]
        tethered = by_arm["TETHERED-2B"][topic_id]
        if (
            set(baseline.accepted_union) != set(tethered.accepted_union)
            or baseline.rrf != tethered.rrf
            or baseline.dual != tethered.dual
        ):
            raise ValueError("arms must reuse identical accepted union, RRF, and DUAL rankings")
        facet_signature = lambda facet: (
            set(facet.scores),
            dict(facet.bm25_ranks),
            dict(facet.text_sha256),
        )
        baseline_facets = {
            (facet.facet_id, facet.manifest_order): facet_signature(facet)
            for facet in baseline.facets
        }
        tethered_facets = {
            (facet.facet_id, facet.manifest_order): facet_signature(facet)
            for facet in tethered.facets
        }
        if baseline_facets != tethered_facets:
            raise ValueError("arms must have identical facet/document pair coverage")
        if any(
            facet.score_schema_version != TASK2_SCORE_SCHEMA_VERSION
            for facet in tethered.facets
        ):
            raise ValueError("TETHERED-2B requires the exact Task 2 score schema")


def _ranking_parameters() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "arms": list(ARMS),
        "topic_ids": list(PILOT_TOPIC_IDS),
        "head_size": HEAD_SIZE,
        "basket_size": BASKET_SIZE,
        "prefix_depths": list(PREFIX_DEPTHS),
        "facet_edge_order": [
            "percentile_desc",
            "manifest_order",
            "bm25_rank",
            "document_id",
        ],
        "raw_scores_cross_query_boundaries": False,
        "tail_order": "prior_DUAL",
        "qrels_opened": False,
    }


def _topic_summary(
    by_arm: Mapping[str, Mapping[str, TopicInput]],
) -> dict[str, dict[str, object]]:
    return {
        topic_id: {
            "accepted_union_count": len(by_arm["FACET-2B"][topic_id].accepted_union),
            "facet_counts": {
                arm: len(by_arm[arm][topic_id].facets) for arm in ARMS
            },
            "duplicate_skip_totals": {
                arm: dict(
                    build_two_basket_permutation(
                        by_arm[arm][topic_id]
                    ).duplicate_skip_totals
                )
                for arm in ARMS
            },
        }
        for topic_id in PILOT_TOPIC_IDS
    }


def freeze_rankings(
    *,
    facet_topics: Sequence[TopicInput],
    tethered_topics: Sequence[TopicInput],
    input_paths: Mapping[str, Path],
    output: Path,
) -> dict[str, object]:
    """Create and seal deterministic FACET-2B and TETHERED-2B artifacts."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"create-only freeze output already exists: {output}")
    by_arm = {
        "FACET-2B": {topic.topic_id: topic for topic in facet_topics},
        "TETHERED-2B": {topic.topic_id: topic for topic in tethered_topics},
    }
    if (
        len(by_arm["FACET-2B"]) != len(facet_topics)
        or len(by_arm["TETHERED-2B"]) != len(tethered_topics)
    ):
        raise ValueError("duplicate protected topic input")
    topic_ids = sorted(by_arm["FACET-2B"], key=PILOT_TOPIC_IDS.index)
    if (
        tuple(topic_ids) != PILOT_TOPIC_IDS
        or set(topic_ids) != set(by_arm["TETHERED-2B"])
    ):
        raise ValueError("both arms must contain the exact protected pilot topics")
    if not input_paths:
        raise ValueError("authenticated input paths are required")
    _validate_arm_contract(by_arm)
    rows: list[dict[str, object]] = []
    prefixes: dict[str, dict[str, dict[str, list[str]]]] = {}
    for topic_id in topic_ids:
        baseline = by_arm["FACET-2B"][topic_id]
        prefixes[topic_id] = {}
        for arm in ARMS:
            result = build_two_basket_permutation(by_arm[arm][topic_id])
            prefixes[topic_id][arm] = {
                str(depth): list(result.document_ids[:depth]) for depth in PREFIX_DEPTHS
            }
            rows.extend(
                _entry_row(topic_id, arm, rank, entry)
                for rank, entry in enumerate(result.entries, 1)
            )
    topic_summary = _topic_summary(by_arm)
    parameters = _ranking_parameters()
    bindings = {
        "schema_version": SCHEMA_VERSION,
        "qrels_opened": False,
        "inputs": {name: _binding(path) for name, path in sorted(input_paths.items())},
        "topic_inputs": {
            arm: {
                topic_id: _topic_payload(by_arm[arm][topic_id])
                for topic_id in topic_ids
            }
            for arm in ARMS
        },
    }
    artifact_bytes = {
        "parameters.json": _pretty_bytes(parameters),
        "input_bindings.json": _pretty_bytes(bindings),
        "rankings.jsonl": _jsonl_bytes(rows),
        "prefixes.json": _pretty_bytes(prefixes),
    }
    output.mkdir(parents=True)
    for name, content in artifact_bytes.items():
        _exclusive_write(output / name, content)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "rankings_frozen_before_qrels",
        "qrels_opened": False,
        "arms": list(ARMS),
        "topic_ids": topic_ids,
        "ranking_row_count": len(rows),
        "topic_summary": topic_summary,
        "artifacts": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in artifact_bytes.items()
        },
    }
    summary_bytes = _pretty_bytes(summary)
    _exclusive_write(output / "summary.json", summary_bytes)
    sealed_files = {
        **artifact_bytes,
        "summary.json": summary_bytes,
    }
    seal_material = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "sealed_before_qrels",
        "qrels_opened": False,
        "files": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in sorted(sealed_files.items())
        },
    }
    seal = {**seal_material, "root_sha256": _sha256(_canonical_bytes(seal_material))}
    _exclusive_write(output / "SEALED.json", _pretty_bytes(seal))
    verify_freeze(output)
    return summary


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def verify_freeze(output: Path) -> dict[str, object]:
    """Recompute every file/input hash and enforce two-basket semantics."""

    output = Path(output)
    expected_names = {
        "parameters.json",
        "input_bindings.json",
        "rankings.jsonl",
        "prefixes.json",
        "summary.json",
        "SEALED.json",
    }
    entries = list(output.iterdir())
    actual_names = {path.name for path in entries}
    if actual_names != expected_names or not all(path.is_file() for path in entries):
        raise ValueError("freeze has missing or extra files")
    seal = _read_object(output / "SEALED.json", "seal")
    seal_material = {key: seal.get(key) for key in ("schema_version", "status", "qrels_opened", "files")}
    if (
        seal_material["schema_version"] != SEAL_SCHEMA_VERSION
        or seal_material["status"] != "sealed_before_qrels"
        or seal_material["qrels_opened"] is not False
        or seal.get("root_sha256") != _sha256(_canonical_bytes(seal_material))
        or not isinstance(seal_material["files"], Mapping)
    ):
        raise ValueError("seal SHA-256 or contract differs")
    for name in sorted(expected_names - {"SEALED.json"}):
        content = (output / name).read_bytes()
        if seal_material["files"].get(name) != {  # type: ignore[union-attr]
            "bytes": len(content),
            "sha256": _sha256(content),
        }:
            raise ValueError(f"artifact SHA-256 differs: {name}")
    summary = _read_object(output / "summary.json", "summary")
    parameters = _read_object(output / "parameters.json", "parameters")
    bindings = _read_object(output / "input_bindings.json", "input bindings")
    prefixes = _read_object(output / "prefixes.json", "prefixes")
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "rankings_frozen_before_qrels"
        or summary.get("qrels_opened") is not False
        or summary.get("arms") != list(ARMS)
        or summary.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or parameters != _ranking_parameters()
        or bindings.get("schema_version") != SCHEMA_VERSION
        or bindings.get("qrels_opened") is not False
    ):
        raise ValueError("freeze semantic contract differs")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("summary artifact records are missing")
    for name in ("parameters.json", "input_bindings.json", "rankings.jsonl", "prefixes.json"):
        content = (output / name).read_bytes()
        if artifacts.get(name) != {"bytes": len(content), "sha256": _sha256(content)}:
            raise ValueError(f"summary SHA-256 differs: {name}")
    input_rows = bindings.get("inputs")
    if not isinstance(input_rows, Mapping) or not input_rows:
        raise ValueError("input binding records are missing")
    for name, raw in input_rows.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"input binding is invalid: {name}")
        path = Path(str(raw.get("path")))
        content = path.read_bytes()
        if raw != {"path": str(path), "bytes": len(content), "sha256": _sha256(content)}:
            raise ValueError(f"input SHA-256 differs: {name}")

    raw_topic_inputs = bindings.get("topic_inputs")
    if not isinstance(raw_topic_inputs, Mapping) or set(raw_topic_inputs) != set(ARMS):
        raise ValueError("frozen semantic inputs are missing")
    semantic_inputs: dict[str, dict[str, TopicInput]] = {}
    for arm in ARMS:
        raw_arm = raw_topic_inputs.get(arm)
        if not isinstance(raw_arm, Mapping) or set(raw_arm) != set(PILOT_TOPIC_IDS):
            raise ValueError("frozen semantic topic population differs")
        semantic_inputs[arm] = {
            topic_id: _topic_from_payload(raw_arm[topic_id])
            for topic_id in PILOT_TOPIC_IDS
        }
    _validate_arm_contract(semantic_inputs)
    if summary.get("topic_summary") != _topic_summary(semantic_inputs):
        raise ValueError("freeze semantic contract differs from frozen inputs")
    expected_rows: list[dict[str, object]] = []
    expected_prefixes: dict[str, dict[str, dict[str, list[str]]]] = {}
    for topic_id in PILOT_TOPIC_IDS:
        expected_prefixes[topic_id] = {}
        for arm in ARMS:
            result = build_two_basket_permutation(semantic_inputs[arm][topic_id])
            expected_rows.extend(
                _entry_row(topic_id, arm, rank, entry)
                for rank, entry in enumerate(result.entries, 1)
            )
            expected_prefixes[topic_id][arm] = {
                str(depth): list(result.document_ids[:depth]) for depth in PREFIX_DEPTHS
            }
    if _jsonl_bytes(expected_rows) != (output / "rankings.jsonl").read_bytes():
        raise ValueError("rankings differ from frozen semantic inputs")
    if prefixes != expected_prefixes:
        raise ValueError("prefixes differ from frozen semantic inputs")

    ranking_bytes = (output / "rankings.jsonl").read_bytes()
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(ranking_bytes.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"rankings line {line_number} is invalid") from exc
        if not isinstance(row, dict):
            raise ValueError("ranking rows must be objects")
        rows.append(row)
    if summary.get("ranking_row_count") != len(rows):
        raise ValueError("ranking population differs from summary")
    topic_summary = summary.get("topic_summary")
    if not isinstance(topic_summary, Mapping) or set(topic_summary) != set(PILOT_TOPIC_IDS):
        raise ValueError("protected topic summary population differs")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        topic_id, arm = str(row.get("topic_id")), str(row.get("arm"))
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or topic_id not in summary.get("topic_ids", [])
            or arm not in ARMS
        ):
            raise ValueError("ranking row identity is invalid")
        grouped[(topic_id, arm)].append(row)
    for topic_id in summary.get("topic_ids", []):
        arm_documents: dict[str, list[str]] = {}
        arm_rrf: dict[str, dict[str, int]] = {}
        arm_dual: dict[str, dict[str, int]] = {}
        for arm in ARMS:
            arm_rows = grouped.get((str(topic_id), arm), [])
            ranks = [row.get("rank") for row in arm_rows]
            documents = [str(row.get("document_id")) for row in arm_rows]
            sources = [row.get("source") for row in arm_rows]
            rrf_ranks = [row.get("rrf_rank") for row in arm_rows]
            dual_ranks = [row.get("dual_rank") for row in arm_rows]
            population_size = len(arm_rows)
            if (
                ranks != list(range(1, len(arm_rows) + 1))
                or len(documents) != len(set(documents))
                or sources[:HEAD_SIZE] != ["rrf_head"] * HEAD_SIZE
                or sources[HEAD_SIZE:INTERLEAVED_END:2] != ["rrf_basket"] * BASKET_SIZE
                or sources[HEAD_SIZE + 1:INTERLEAVED_END:2] != ["facet_basket"] * BASKET_SIZE
                or sources[INTERLEAVED_END:] != ["dual_tail"] * (len(sources) - INTERLEAVED_END)
                or set(rrf_ranks) != set(range(1, population_size + 1))
                or set(dual_ranks) != set(range(1, population_size + 1))
                or rrf_ranks[:HEAD_SIZE] != list(range(1, HEAD_SIZE + 1))
                or rrf_ranks[HEAD_SIZE:INTERLEAVED_END:2]
                != list(range(HEAD_SIZE + 1, HEAD_SIZE + BASKET_SIZE + 1))
                or dual_ranks[INTERLEAVED_END:]
                != sorted(dual_ranks[INTERLEAVED_END:])
            ):
                raise ValueError("two-basket semantic invariant differs")
            topic_record = topic_summary.get(str(topic_id))
            if (
                not isinstance(topic_record, Mapping)
                or topic_record.get("accepted_union_count") != population_size
                or not isinstance(topic_record.get("facet_counts"), Mapping)
                or topic_record["facet_counts"].get(arm, 0) <= 0  # type: ignore[index]
            ):
                raise ValueError("protected topic population differs")
            rrf_basket = set(documents[HEAD_SIZE:INTERLEAVED_END:2])
            facet_basket = set(documents[HEAD_SIZE + 1:INTERLEAVED_END:2])
            if rrf_basket & facet_basket:
                raise ValueError("RRF and facet baskets are not disjoint")
            for row in arm_rows:
                if row["source"] == "facet_basket":
                    provenance = row.get("score_provenance")
                    if (
                        not row.get("generating_facet")
                        or not _valid_sha256(row.get("query_sha256"))
                        or not _valid_sha256(row.get("text_sha256"))
                        or not isinstance(provenance, Mapping)
                        or provenance.get("score_space") != "raw_query_local_logits"
                        or not 0 < float(row.get("percentile", 0)) <= 1
                    ):
                        raise ValueError("facet score provenance is invalid")
            arm_documents[arm] = documents
            arm_rrf[arm] = dict(zip(documents, rrf_ranks, strict=True))
            arm_dual[arm] = dict(zip(documents, dual_ranks, strict=True))
            expected_prefixes = {
                str(depth): documents[:depth] for depth in PREFIX_DEPTHS
            }
            if prefixes.get(str(topic_id), {}).get(arm) != expected_prefixes:  # type: ignore[union-attr]
                raise ValueError("frozen prefixes differ from rankings")
        if set(arm_documents[ARMS[0]]) != set(arm_documents[ARMS[1]]):
            raise ValueError("arm accepted-union populations differ")
        if arm_documents[ARMS[0]][:HEAD_SIZE] != arm_documents[ARMS[1]][:HEAD_SIZE]:
            raise ValueError("arms do not preserve identical RRF heads")
        if arm_rrf[ARMS[0]] != arm_rrf[ARMS[1]] or arm_dual[ARMS[0]] != arm_dual[ARMS[1]]:
            raise ValueError("arms do not reuse identical RRF and DUAL rankings")
    return summary
