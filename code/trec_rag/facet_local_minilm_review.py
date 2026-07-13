"""Qrels-blind, system-masked facet relevance review for the MiniLM pilot."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .facet_local_minilm_manifest import PROTECTED_TOPIC_IDS
from .facet_local_minilm_rank import verify_freeze


CONTROL_ARM = "C0_TOPIC_LOCAL"
MINILM_ARM = "BF50_TOPIC_LOCAL"
LABEL_VOCABULARY = {
    "direct_answer",
    "partial_or_related",
    "not_facet_relevant",
}
PILOT_TOPIC_IDS = ("200", "225", "707", "897")
EXPECTED_FACET_COUNTS = {"200": 9, "225": 7, "707": 3, "897": 8}

RUBRIC_TEXT = """# Facet relevance review

Judge only whether the **displayed passage** answers the facet query. Do not
infer relevance from an unseen document or from outside knowledge.

- `direct_answer`: the displayed passage directly supplies requested facet information.
- `partial_or_related`: it is on-domain and useful, but only partially answers the facet.
- `not_facet_relevant`: it does not answer the facet.
- `wrong_domain`: flag an answer about the wrong population or subject domain.
- `low_quality`: flag dictionaries, essay templates, spam, or similarly weak material.

Write one JSON object per line with exactly:
`item_id`, `relevance`, `wrong_domain` (boolean), `low_quality` (boolean), and
`reviewer_id`. Example:
`{"item_id":"…","low_quality":false,"relevance":"direct_answer","reviewer_id":"reviewer-a","wrong_domain":false}`
"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line:
            raise ValueError(f"{path.name} row {number} is empty")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} row {number} must be an object")
        rows.append(value)
    return rows


def _exclusive_write(path: Path, source: bytes, mode: int = 0o644) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(source)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory_noreplace(stage: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, "atomic no-replace publication unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(stage), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, "create-only destination exists", destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _retry_safe_leaf(path: Path, source: bytes, mode: int = 0o600) -> None:
    if path.exists():
        if path.read_bytes() == source:
            return
        raise FileExistsError(f"immutable review artifact differs: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{_sha256_bytes(source)[:12]}")
    try:
        _exclusive_write(temporary, source, mode)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != source:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def _reject_protected(topic_ids: Iterable[object]) -> None:
    for raw in topic_ids:
        topic_id = str(raw)
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")


def _ranked(rows: Sequence[Mapping[str, object]], *, label: str) -> list[dict[str, object]]:
    result = [dict(row) for row in rows]
    result.sort(key=lambda row: (int(row.get("rank", 0)), str(row.get("document_id", ""))))
    if [int(row.get("rank", 0)) for row in result] != list(range(1, len(result) + 1)):
        raise ValueError(f"{label} ranks must be consecutive from 1")
    if len({str(row.get("document_id", "")) for row in result}) != len(result):
        raise ValueError(f"{label} document IDs must be unique")
    return result


def _passage_provenance(
    stream: Mapping[str, object], row: Mapping[str, object]
) -> tuple[str, dict[str, object]]:
    selected = row.get("selected_windows")
    if not isinstance(selected, list) or not selected or not isinstance(selected[0], Mapping):
        raise ValueError("MiniLM top4 row lacks selected window provenance")
    window = selected[0]
    passage = str(window.get("window_text", ""))
    if not passage or passage != str(row.get("passage", "")):
        raise ValueError("MiniLM displayed passage differs from highest-scoring window")
    provenance = {
        "aggregation": "top4",
        "document_end_token": int(window.get("document_end_token", 0)),
        "document_start_token": int(window.get("document_start_token", 0)),
        "minilm_source_rank": int(row.get("rank", 0)),
        "passage_sha256": _sha256_text(passage),
        "query_sha256": str(row.get("query_sha256", "")),
        "raw_score": float(window.get("score", 0.0)),
        "selection": "highest_raw_logit_window",
        "stream_path": str(stream.get("top4_path", "")),
        "stream_sha256": str(stream.get("top4_sha256", "")),
        "window_id": str(window.get("window_id", "")),
    }
    return passage, provenance


def build_review_packet(
    ranking_freeze: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Pool top-two BM25/MiniLM facet rows and mask system identity."""

    topics = [str(value) for value in ranking_freeze.get("topic_ids", [])]
    streams = ranking_freeze.get("facet_streams")
    if not isinstance(streams, list):
        raise ValueError("facet streams are required")
    # Reject before any stream join or row traversal.
    _reject_protected(topics)
    _reject_protected(
        stream.get("topic_id")
        for stream in streams
        if isinstance(stream, Mapping)
    )
    if tuple(sorted(topics)) != tuple(sorted(PILOT_TOPIC_IDS)):
        raise ValueError("review topics must be the four pilot topics")

    items: dict[tuple[str, str, str], dict[str, object]] = {}
    facets: set[tuple[str, str]] = set()
    membership_count = 0
    for stream in sorted(
        streams,
        key=lambda row: (str(row.get("topic_id", "")), str(row.get("variant_name", ""))),
    ):
        if not isinstance(stream, Mapping):
            raise ValueError("facet stream must be an object")
        topic_id = str(stream.get("topic_id", ""))
        variant = str(stream.get("variant_name", ""))
        if stream.get("family") != "facet":
            raise ValueError("review stream must be a facet")
        facets.add((topic_id, variant))
        bm25 = _ranked(stream.get("bm25", []), label="BM25 stream")  # type: ignore[arg-type]
        top4 = _ranked(stream.get("top4", []), label="MiniLM stream")  # type: ignore[arg-type]
        if not bm25 or not top4:
            raise ValueError("facet review streams cannot be empty")
        query = str(top4[0].get("query", ""))
        query_sha = str(top4[0].get("query_sha256", ""))
        top4_by_doc = {str(row["document_id"]): row for row in top4}
        if set(top4_by_doc) != {str(row["document_id"]) for row in bm25}:
            raise ValueError("BM25 and MiniLM facet streams must contain the same documents")
        for arm, rows in ((CONTROL_ARM, bm25[:2]), (MINILM_ARM, top4[:2])):
            for row in rows:
                document_id = str(row["document_id"])
                minilm_row = top4_by_doc[document_id]
                if (
                    str(minilm_row.get("query", "")) != query
                    or str(minilm_row.get("query_sha256", "")) != query_sha
                ):
                    raise ValueError("facet query lineage mismatch")
                passage, provenance = _passage_provenance(stream, minilm_row)
                key = (topic_id, variant, document_id)
                item_id = _sha256_text(
                    "\n".join(
                        (
                            str(ranking_freeze.get("experiment_manifest_sha256", "")),
                            topic_id,
                            variant,
                            document_id,
                        )
                    )
                )
                item = items.setdefault(
                    key,
                    {
                        "document_id": document_id,
                        "facet_query": query,
                        "item_id": item_id,
                        "memberships": [],
                        "passage": passage,
                        "topic_id": topic_id,
                    },
                )
                if item["passage"] != passage or item["facet_query"] != query:
                    raise ValueError("shared review item has inconsistent passage/query")
                membership = {
                    "arm": arm,
                    "document_id": document_id,
                    "facet": {"topic_id": topic_id, "variant_name": variant},
                    "passage_selection_provenance": provenance,
                    "source_rank": int(row["rank"]),
                }
                memberships = item["memberships"]
                assert isinstance(memberships, list)
                if any(
                    existing["arm"] == arm and existing["facet"] == membership["facet"]
                    for existing in memberships
                ):
                    raise ValueError("duplicate within-facet review membership")
                memberships.append(membership)
                membership_count += 1

    counts = defaultdict(int)
    for topic_id, _variant in facets:
        counts[topic_id] += 1
    if dict(counts) != EXPECTED_FACET_COUNTS or len(facets) != 27:
        raise ValueError("review requires exact 27-facet distribution")
    if membership_count != 108:
        raise ValueError("review requires exactly 108 arm/facet memberships")

    secret_items = sorted(items.values(), key=lambda item: str(item["item_id"]))
    for item in secret_items:
        memberships = item["memberships"]
        assert isinstance(memberships, list)
        memberships.sort(
            key=lambda row: (
                str(row["arm"]),
                int(row["source_rank"]),
            )
        )
    packet = [
        {
            "facet_query": item["facet_query"],
            "item_id": item["item_id"],
            "passage": item["passage"],
            "topic_id": item["topic_id"],
        }
        for item in secret_items
    ]
    seed = str(ranking_freeze.get("experiment_manifest_sha256", ""))
    random.Random(int(seed[:16], 16)).shuffle(packet)
    secret = {
        "experiment_manifest_sha256": seed,
        "items": secret_items,
        "packet_sha256": _sha256_bytes(_jsonl_bytes(packet)),
        "ranking_freeze_sha256": str(ranking_freeze.get("ranking_freeze_sha256", "")),
        "schema_version": "facet-local-minilm-review-secret-v1",
        "topic_ids": list(PILOT_TOPIC_IDS),
    }
    return packet, secret


def _validate_packet_secret(
    packet: Sequence[Mapping[str, object]], secret: Mapping[str, object]
) -> tuple[int, int]:
    _reject_protected(row.get("topic_id") for row in packet)
    if any(set(row) != {"item_id", "topic_id", "facet_query", "passage"} for row in packet):
        raise ValueError("public packet schema mismatch")
    if secret.get("schema_version") != "facet-local-minilm-review-secret-v1":
        raise ValueError("secret map schema mismatch")
    if secret.get("packet_sha256") != _sha256_bytes(_jsonl_bytes(packet)):
        raise ValueError("secret map packet binding mismatch")
    items = secret.get("items")
    if not isinstance(items, list):
        raise ValueError("secret map items are invalid")
    _reject_protected(item.get("topic_id") for item in items if isinstance(item, Mapping))
    packet_by_id = {str(row["item_id"]): dict(row) for row in packet}
    secret_by_id = {
        str(item.get("item_id", "")): item for item in items if isinstance(item, Mapping)
    }
    if len(packet_by_id) != len(packet) or len(secret_by_id) != len(items):
        raise ValueError("review item IDs must be unique")
    if set(packet_by_id) != set(secret_by_id):
        raise ValueError("packet and secret item IDs differ")
    facets: set[tuple[str, str]] = set()
    membership_count = 0
    for item_id, item in secret_by_id.items():
        public = packet_by_id[item_id]
        if (
            public["topic_id"] != item.get("topic_id")
            or public["facet_query"] != item.get("facet_query")
            or public["passage"] != item.get("passage")
        ):
            raise ValueError("packet differs from secret public projection")
        if item.get("document_id") in {None, ""}:
            raise ValueError("secret document ID is invalid")
        memberships = item.get("memberships")
        if not isinstance(memberships, list) or not memberships:
            raise ValueError("secret memberships are invalid")
        seen: set[tuple[str, str, str]] = set()
        for membership in memberships:
            if not isinstance(membership, Mapping):
                raise ValueError("secret membership must be an object")
            facet = membership.get("facet")
            provenance = membership.get("passage_selection_provenance")
            if not isinstance(facet, Mapping) or not isinstance(provenance, Mapping):
                raise ValueError("secret membership lineage is invalid")
            topic_id = str(facet.get("topic_id", ""))
            variant = str(facet.get("variant_name", ""))
            _reject_protected((topic_id,))
            if topic_id != item.get("topic_id") or membership.get("document_id") != item.get("document_id"):
                raise ValueError("secret membership item lineage mismatch")
            arm = str(membership.get("arm", ""))
            if arm not in {CONTROL_ARM, MINILM_ARM} or membership.get("source_rank") not in {1, 2}:
                raise ValueError("secret membership arm/rank is invalid")
            if provenance.get("selection") != "highest_raw_logit_window":
                raise ValueError("secret passage selection is invalid")
            if provenance.get("passage_sha256") != _sha256_text(str(item.get("passage", ""))):
                raise ValueError("secret passage hash mismatch")
            identity = (arm, topic_id, variant)
            if identity in seen:
                raise ValueError("duplicate within-facet membership")
            seen.add(identity)
            facets.add((topic_id, variant))
            membership_count += 1
    counts = defaultdict(int)
    for topic_id, _variant in facets:
        counts[topic_id] += 1
    if dict(counts) != EXPECTED_FACET_COUNTS or membership_count != 108:
        raise ValueError("secret facet/membership accounting mismatch")
    return len(facets), membership_count


def _validate_label_freeze(label_freeze: Mapping[str, object]) -> None:
    if label_freeze.get("status") != "labels_and_adjudication_frozen":
        raise ValueError("labels must be frozen before unmasking")
    required = {
        "adjudicated_disagreement_count",
        "items",
        "label_freeze_sha256",
        "packet_sha256",
        "reviewer_count",
        "schema_version",
        "source_hashes",
        "status",
    }
    if set(label_freeze) != required:
        raise ValueError("label freeze fields mismatch")
    if (
        label_freeze.get("schema_version") != "facet-local-minilm-label-freeze-v1"
        or label_freeze.get("status") != "labels_and_adjudication_frozen"
    ):
        raise ValueError("labels must be frozen before unmasking")
    without_hash = {key: value for key, value in label_freeze.items() if key != "label_freeze_sha256"}
    if label_freeze.get("label_freeze_sha256") != _sha256_bytes(_pretty_bytes(without_hash)):
        raise ValueError("label freeze self-hash mismatch")
    sources = label_freeze.get("source_hashes")
    if not isinstance(sources, Mapping) or set(sources) != {
        "labels_a_sha256",
        "labels_b_sha256",
        "adjudication_sha256",
    } or not all(isinstance(value, str) and len(value) == 64 for value in sources.values()):
        raise ValueError("label freeze source hashes are invalid")
    items = label_freeze.get("items")
    if not isinstance(items, list) or any(
        not isinstance(row, Mapping)
        or set(row) != {"item_id", "low_quality", "relevance", "wrong_domain"}
        or row.get("relevance") not in LABEL_VOCABULARY
        or not isinstance(row.get("low_quality"), bool)
        or not isinstance(row.get("wrong_domain"), bool)
        for row in items
    ):
        raise ValueError("label freeze items are invalid")


def _index_labels(
    packet: Sequence[Mapping[str, object]],
    labels: Sequence[Mapping[str, object]],
    *,
    name: str,
) -> tuple[dict[str, dict[str, object]], str]:
    packet_ids = [str(row.get("item_id", "")) for row in packet]
    if len(packet_ids) != len(set(packet_ids)):
        raise ValueError("packet item IDs must be unique")
    if len(labels) != len(packet_ids):
        raise ValueError(f"{name} must label every packet item")
    result: dict[str, dict[str, object]] = {}
    reviewers: set[str] = set()
    for raw in labels:
        row = dict(raw)
        if set(row) != {"item_id", "low_quality", "relevance", "reviewer_id", "wrong_domain"}:
            raise ValueError(f"{name} label schema mismatch")
        item_id = str(row["item_id"])
        relevance = row["relevance"]
        reviewer = str(row["reviewer_id"])
        if relevance not in LABEL_VOCABULARY:
            raise ValueError("label vocabulary mismatch")
        if not reviewer or not isinstance(row["low_quality"], bool) or not isinstance(row["wrong_domain"], bool):
            raise ValueError(f"{name} label values are invalid")
        if item_id in result:
            raise ValueError(f"{name} contains duplicate item IDs")
        result[item_id] = row
        reviewers.add(reviewer)
    if set(result) != set(packet_ids) or len(reviewers) != 1:
        raise ValueError(f"{name} labels/reviewer are incomplete")
    return result, next(iter(reviewers))


def freeze_labels(
    packet: Sequence[Mapping[str, object]],
    labels_a: Sequence[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]],
    adjudication: Sequence[Mapping[str, object]] | Mapping[str, object],
    *,
    _source_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Freeze labels without exposing system or document identity."""

    _reject_protected(row.get("topic_id") for row in packet)
    a, reviewer_a = _index_labels(packet, labels_a, name="reviewer A")
    b, reviewer_b = _index_labels(packet, labels_b, name="reviewer B")
    if reviewer_a == reviewer_b:
        raise ValueError("two independent reviewers are required")
    disagreements = {
        item_id for item_id in a if a[item_id]["relevance"] != b[item_id]["relevance"]
    }
    adjudication_rows = list(adjudication) if not isinstance(adjudication, Mapping) else []
    adjudicated: dict[str, dict[str, object]] = {}
    adjudicators: set[str] = set()
    for raw in adjudication_rows:
        row = dict(raw)
        if set(row) != {"item_id", "relevance", "reviewer_id"}:
            raise ValueError("adjudication schema mismatch")
        item_id = str(row["item_id"])
        if item_id not in disagreements or row["relevance"] not in LABEL_VOCABULARY:
            raise ValueError("adjudication must cover only disagreements")
        reviewer = str(row["reviewer_id"])
        if reviewer in {reviewer_a, reviewer_b}:
            raise ValueError("adjudication requires an independent third reviewer")
        if item_id in adjudicated:
            raise ValueError("duplicate adjudication")
        adjudicated[item_id] = row
        adjudicators.add(reviewer)
    if set(adjudicated) != disagreements:
        raise ValueError("unadjudicated disagreement remains")
    if disagreements and len(adjudicators) != 1:
        raise ValueError("adjudication requires one third reviewer")

    frozen_items: list[dict[str, object]] = []
    for packet_row in packet:
        item_id = str(packet_row["item_id"])
        relevance = (
            adjudicated[item_id]["relevance"] if item_id in disagreements else a[item_id]["relevance"]
        )
        frozen_items.append(
            {
                "item_id": item_id,
                "low_quality": bool(a[item_id]["low_quality"] or b[item_id]["low_quality"]),
                "relevance": relevance,
                "wrong_domain": bool(a[item_id]["wrong_domain"] or b[item_id]["wrong_domain"]),
            }
        )
    source_hashes = dict(
        _source_hashes
        or {
            "labels_a_sha256": _sha256_bytes(_jsonl_bytes(labels_a)),
            "labels_b_sha256": _sha256_bytes(_jsonl_bytes(labels_b)),
            "adjudication_sha256": _sha256_bytes(_jsonl_bytes(adjudication_rows)),
        }
    )
    if set(source_hashes) != {
        "labels_a_sha256",
        "labels_b_sha256",
        "adjudication_sha256",
    }:
        raise ValueError("label source hash fields mismatch")
    payload: dict[str, object] = {
        "adjudicated_disagreement_count": len(disagreements),
        "items": sorted(frozen_items, key=lambda row: str(row["item_id"])),
        "packet_sha256": _sha256_bytes(_jsonl_bytes(packet)),
        "reviewer_count": 2 + bool(disagreements),
        "schema_version": "facet-local-minilm-label-freeze-v1",
        "source_hashes": source_hashes,
        "status": "labels_and_adjudication_frozen",
    }
    payload["label_freeze_sha256"] = _sha256_bytes(_pretty_bytes(payload))
    return payload


def unmask_review(
    label_freeze: Mapping[str, object], secret: Mapping[str, object]
) -> dict[str, object]:
    _validate_label_freeze(label_freeze)
    labels_raw = label_freeze.get("items")
    secret_raw = secret.get("items")
    if not isinstance(labels_raw, list) or not isinstance(secret_raw, list):
        raise ValueError("label/secret items are invalid")
    _reject_protected(
        item.get("topic_id") for item in secret_raw if isinstance(item, Mapping)
    )
    if secret.get("packet_sha256") != label_freeze.get("packet_sha256"):
        raise ValueError("label freeze packet binding differs from secret map")
    labels = {str(row["item_id"]): dict(row) for row in labels_raw}
    secret_items = {str(row["item_id"]): dict(row) for row in secret_raw}
    if set(labels) != set(secret_items):
        raise ValueError("label freeze and secret map item IDs differ")

    by_arm: dict[str, dict[str, int]] = {}
    by_facet: dict[tuple[str, str, str], dict[str, int]] = {}
    unmasked: list[dict[str, object]] = []
    metric_names = sorted(LABEL_VOCABULARY) + ["wrong_domain", "low_quality"]

    def counter() -> dict[str, int]:
        return {"denominator": 0, **{name: 0 for name in metric_names}}

    for item_id in sorted(labels):
        label = labels[item_id]
        secret_item = secret_items[item_id]
        memberships = secret_item.get("memberships")
        if not isinstance(memberships, list):
            raise ValueError("secret memberships are invalid")
        unmasked.append({**secret_item, "label": label})
        for membership in memberships:
            arm = str(membership["arm"])
            facet = membership["facet"]
            key = (arm, str(facet["topic_id"]), str(facet["variant_name"]))
            for target in (by_arm.setdefault(arm, counter()), by_facet.setdefault(key, counter())):
                target["denominator"] += 1
                target[str(label["relevance"])] += 1
                target["wrong_domain"] += int(bool(label["wrong_domain"]))
                target["low_quality"] += int(bool(label["low_quality"]))

    arm_rows = [{"arm": arm, **by_arm[arm]} for arm in sorted(by_arm)]
    facet_rows = [
        {"arm": key[0], "topic_id": key[1], "variant_name": key[2], **by_facet[key]}
        for key in sorted(by_facet)
    ]
    return {
        "aggregate_counts": {"by_arm": arm_rows, "by_arm_topic_facet": facet_rows},
        "unmasked_items": unmasked,
    }


def freeze_review(
    packet: Sequence[Mapping[str, object]],
    labels_a: Sequence[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]],
    adjudication: Sequence[Mapping[str, object]] | Mapping[str, object],
    *,
    secret: Mapping[str, object] | None = None,
) -> dict[str, object]:
    label_freeze = freeze_labels(packet, labels_a, labels_b, adjudication)
    if secret is None:
        raise ValueError("secret map is required after labels are frozen")
    unmasked = unmask_review(label_freeze, secret)
    return {
        **unmasked,
        "label_freeze": label_freeze,
        "qrels_opened": False,
        "schema_version": "facet-local-minilm-review-freeze-v1",
        "status": "review_frozen_before_qrels",
    }


def write_review_create(
    output_dir: Path,
    packet: Sequence[Mapping[str, object]],
    secret: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(output_dir)
    _reject_protected(row.get("topic_id") for row in packet)
    facet_count, memberships = _validate_packet_secret(packet, secret)
    if destination.exists() or os.path.lexists(destination):
        raise FileExistsError(f"create-only review destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    packet_bytes = _jsonl_bytes(packet)
    secret_bytes = _pretty_bytes(secret)
    rubric_bytes = RUBRIC_TEXT.encode("utf-8")
    receipt: dict[str, object] = {
        "facet_count": facet_count,
        "inference_call_count": 0,
        "item_count": len(packet),
        "membership_count": memberships,
        "network_call_count": 0,
        "packet_bytes": len(packet_bytes),
        "packet_sha256": _sha256_bytes(packet_bytes),
        "qrels_opened": False,
        "ranking_freeze_sha256": secret.get("ranking_freeze_sha256"),
        "retrieval_call_count": 0,
        "rubric_sha256": _sha256_bytes(rubric_bytes),
        "schema_version": "facet-local-minilm-review-create-receipt-v1",
        "secret_map_sha256": _sha256_bytes(secret_bytes),
        "status": "awaiting_two_independent_reviews",
    }
    try:
        _exclusive_write(stage / "packet.jsonl", packet_bytes)
        _exclusive_write(stage / "secret_map.json", secret_bytes, 0o600)
        _exclusive_write(stage / "rubric.md", rubric_bytes)
        _exclusive_write(stage / "create_receipt.json", _pretty_bytes(receipt))
        _publish_directory_noreplace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return receipt


def verify_review_create(
    review_dir: Path, *, ranking_freeze_dir: Path | None = None
) -> dict[str, object]:
    root = Path(review_dir)
    receipt = _read_json(root / "create_receipt.json")
    packet_source = (root / "packet.jsonl").read_bytes()
    secret_source = (root / "secret_map.json").read_bytes()
    rubric_source = (root / "rubric.md").read_bytes()
    if _sha256_bytes(packet_source) != receipt.get("packet_sha256"):
        raise ValueError("packet hash mismatch")
    if _sha256_bytes(secret_source) != receipt.get("secret_map_sha256"):
        raise ValueError("secret map hash mismatch")
    if _sha256_bytes(rubric_source) != receipt.get("rubric_sha256"):
        raise ValueError("rubric hash mismatch")
    packet = _read_jsonl(root / "packet.jsonl")
    secret = _read_json(root / "secret_map.json")
    facet_count, membership_count = _validate_packet_secret(packet, secret)
    if (root / "secret_map.json").stat().st_mode & 0o777 != 0o600:
        raise ValueError("secret map permissions must be 0600")
    if len(packet) != receipt.get("item_count"):
        raise ValueError("packet row count mismatch")
    if (
        receipt.get("facet_count") != facet_count
        or receipt.get("membership_count") != membership_count
        or receipt.get("qrels_opened") is not False
        or any(
            receipt.get(name) != 0
            for name in ("retrieval_call_count", "network_call_count", "inference_call_count")
        )
    ):
        raise ValueError("create receipt semantic accounting mismatch")
    if ranking_freeze_dir is not None:
        freeze = verify_freeze(Path(ranking_freeze_dir))
        if freeze.get("freeze_sha256") != receipt.get("ranking_freeze_sha256"):
            raise ValueError("ranking freeze hash mismatch")
        expected_packet, expected_secret = build_review_packet(
            _real_review_input(Path(ranking_freeze_dir))
        )
        if packet_source != _jsonl_bytes(expected_packet) or secret_source != _pretty_bytes(
            expected_secret
        ):
            raise ValueError("create stage differs from authenticated ranking freeze")
    return {
        "item_count": len(packet),
        "packet_sha256": receipt["packet_sha256"],
        "status": "verified_create_stage",
    }


def _real_review_input(freeze_dir: Path) -> dict[str, object]:
    root = Path(freeze_dir)
    verified = verify_freeze(root)
    freeze = _read_json(root / "freeze.json")
    if verified.get("freeze_sha256") != freeze.get("freeze_sha256"):
        raise ValueError("ranking freeze verification summary mismatch")
    manifest = _read_json(root / "stream_manifest.json")
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for entry in manifest.get("streams", []):  # type: ignore[union-attr]
        if not isinstance(entry, Mapping) or entry.get("family") != "facet":
            continue
        if entry.get("aggregation") not in {"bm25", "top4"}:
            continue
        topic_id = str(entry["topic_id"])
        _reject_protected((topic_id,))
        variant = str(entry["variant_name"])
        key = (topic_id, variant)
        stream = grouped.setdefault(
            key,
            {
                "family": "facet",
                "retriever_name": entry["retriever_name"],
                "topic_id": topic_id,
                "variant_name": variant,
            },
        )
        aggregation = str(entry["aggregation"])
        path = root / str(entry["path"])
        if _sha256_bytes(path.read_bytes()) != entry.get("file_sha256"):
            raise ValueError("review stream hash mismatch")
        stream[aggregation] = _read_jsonl(path)
        stream[f"{aggregation}_path"] = str(entry["path"])
        stream[f"{aggregation}_sha256"] = str(entry["file_sha256"])
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("ranking freeze bindings are invalid")
    return {
        "experiment_manifest_sha256": bindings["manifest_sha256"],
        "facet_streams": list(grouped.values()),
        "ranking_freeze_sha256": freeze["freeze_sha256"],
        "topic_ids": freeze["topic_ids"],
    }


def create_review(ranking_freeze_dir: Path, output_dir: Path) -> dict[str, object]:
    packet, secret = build_review_packet(_real_review_input(Path(ranking_freeze_dir)))
    return write_review_create(Path(output_dir), packet, secret)


def freeze_review_directory(
    review_dir: Path,
    *,
    labels_a_path: Path,
    labels_b_path: Path,
    adjudication_path: Path,
) -> dict[str, object]:
    root = Path(review_dir)
    # Validate public inputs and freeze the independent label sources before
    # opening the private unmask map.
    receipt_source = (root / "create_receipt.json").read_bytes()
    receipt = _read_json(root / "create_receipt.json")
    packet = _read_jsonl(root / "packet.jsonl")
    _reject_protected(row.get("topic_id") for row in packet)
    packet_source = (root / "packet.jsonl").read_bytes()
    if _sha256_bytes(packet_source) != receipt.get("packet_sha256"):
        raise ValueError("packet hash mismatch before label freeze")
    labels_a_source = Path(labels_a_path).read_bytes()
    labels_b_source = Path(labels_b_path).read_bytes()
    adjudication_source = Path(adjudication_path).read_bytes()
    labels_a = _read_jsonl(Path(labels_a_path))
    labels_b = _read_jsonl(Path(labels_b_path))
    adjudication = _read_jsonl(Path(adjudication_path))
    source_hashes = {
        "labels_a_sha256": _sha256_bytes(labels_a_source),
        "labels_b_sha256": _sha256_bytes(labels_b_source),
        "adjudication_sha256": _sha256_bytes(adjudication_source),
    }
    label_freeze = freeze_labels(
        packet,
        labels_a,
        labels_b,
        adjudication,
        _source_hashes=source_hashes,
    )
    _validate_label_freeze(label_freeze)
    _retry_safe_leaf(root / "labels_a.jsonl", labels_a_source)
    _retry_safe_leaf(root / "labels_b.jsonl", labels_b_source)
    _retry_safe_leaf(root / "adjudication.jsonl", adjudication_source)
    verify_review_create(root)
    secret_source = (root / "secret_map.json").read_bytes()
    secret = _read_json(root / "secret_map.json")
    unmasked_result = unmask_review(label_freeze, secret)
    unmasked = unmasked_result["unmasked_items"]
    unmasked_bytes = _jsonl_bytes(unmasked)  # type: ignore[arg-type]
    bindings = {
        "adjudication_sha256": source_hashes["adjudication_sha256"],
        "create_receipt_sha256": _sha256_bytes(receipt_source),
        "label_freeze_sha256": label_freeze["label_freeze_sha256"],
        "labels_a_sha256": source_hashes["labels_a_sha256"],
        "labels_b_sha256": source_hashes["labels_b_sha256"],
        "packet_sha256": receipt["packet_sha256"],
        "ranking_freeze_sha256": receipt["ranking_freeze_sha256"],
        "rubric_sha256": receipt["rubric_sha256"],
        "secret_map_sha256": _sha256_bytes(secret_source),
        "unmasked_items_sha256": _sha256_bytes(unmasked_bytes),
    }
    review_payload: dict[str, object] = {
        "aggregate_counts": unmasked_result["aggregate_counts"],
        "bindings": bindings,
        "qrels_opened": False,
        "schema_version": "facet-local-minilm-review-freeze-v2",
        "status": "review_frozen_before_qrels",
    }
    review_payload["review_freeze_sha256"] = _sha256_bytes(_pretty_bytes(review_payload))
    _retry_safe_leaf(root / "label_freeze.json", _pretty_bytes(label_freeze))
    _retry_safe_leaf(root / "unmasked_items.jsonl", unmasked_bytes)
    # Root commit marker is always published last.
    _retry_safe_leaf(root / "review_freeze.json", _pretty_bytes(review_payload))
    return review_payload


def verify_review_freeze(review_dir: Path) -> dict[str, object]:
    root = Path(review_dir)
    create_verified = verify_review_create(root)
    receipt_source = (root / "create_receipt.json").read_bytes()
    receipt = _read_json(root / "create_receipt.json")
    label_freeze = _read_json(root / "label_freeze.json")
    review_freeze = _read_json(root / "review_freeze.json")
    unmasked_source = (root / "unmasked_items.jsonl").read_bytes()
    _validate_label_freeze(label_freeze)
    if (
        review_freeze.get("schema_version") != "facet-local-minilm-review-freeze-v2"
        or review_freeze.get("status") != "review_frozen_before_qrels"
    ):
        raise ValueError("review freeze status mismatch")
    if review_freeze.get("qrels_opened") is not False:
        raise ValueError("review freeze qrels firewall mismatch")
    without_hash = {
        key: value for key, value in review_freeze.items() if key != "review_freeze_sha256"
    }
    if review_freeze.get("review_freeze_sha256") != _sha256_bytes(
        _pretty_bytes(without_hash)
    ):
        raise ValueError("review freeze self-hash mismatch")
    bindings = review_freeze.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("review freeze bindings are invalid")
    label_sources = label_freeze["source_hashes"]
    expected_bindings = {
        "adjudication_sha256": label_sources["adjudication_sha256"],
        "create_receipt_sha256": _sha256_bytes(receipt_source),
        "label_freeze_sha256": label_freeze["label_freeze_sha256"],
        "labels_a_sha256": label_sources["labels_a_sha256"],
        "labels_b_sha256": label_sources["labels_b_sha256"],
        "packet_sha256": create_verified["packet_sha256"],
        "ranking_freeze_sha256": receipt["ranking_freeze_sha256"],
        "rubric_sha256": receipt["rubric_sha256"],
        "secret_map_sha256": receipt["secret_map_sha256"],
        "unmasked_items_sha256": _sha256_bytes(unmasked_source),
    }
    if dict(bindings) != expected_bindings:
        raise ValueError("review freeze binding mismatch")
    for name, binding_name in (
        ("labels_a.jsonl", "labels_a_sha256"),
        ("labels_b.jsonl", "labels_b_sha256"),
        ("adjudication.jsonl", "adjudication_sha256"),
    ):
        if _sha256_bytes((root / name).read_bytes()) != bindings[binding_name]:
            raise ValueError(f"{name} source hash mismatch")
    secret = _read_json(root / "secret_map.json")
    recomputed = unmask_review(label_freeze, secret)
    if _jsonl_bytes(recomputed["unmasked_items"]) != unmasked_source:
        raise ValueError("unmasked items hash mismatch")
    if recomputed["aggregate_counts"] != review_freeze.get("aggregate_counts"):
        raise ValueError("review aggregate counts mismatch")
    return {
        "review_freeze_sha256": review_freeze["review_freeze_sha256"],
        "status": "verified_review_freeze",
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a masked facet relevance review")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--freeze", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--review", type=Path, required=True)
    freeze.add_argument("--labels-a", type=Path, required=True)
    freeze.add_argument("--labels-b", type=Path, required=True)
    freeze.add_argument("--adjudication", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--review", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.command == "create":
        result = create_review(args.freeze, args.output)
    elif args.command == "freeze":
        result = freeze_review_directory(
            args.review,
            labels_a_path=args.labels_a,
            labels_b_path=args.labels_b,
            adjudication_path=args.adjudication,
        )
    else:
        try:
            result = verify_review_freeze(args.review)
        except FileNotFoundError:
            result = verify_review_create(args.review)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
