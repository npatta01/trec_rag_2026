"""Freeze the qrels-blind full-union adaptive-evidence contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar


PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
PILOT_TOPIC_IDS = ("219", "72", "300", "84")
SCHEMA_VERSION = "adaptive-evidence-contract-v1"

T = TypeVar("T")


def canonical_sha256(value: object) -> str:
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def reject_protected_before_access(
    topic_ids: Iterable[str], loader: Callable[[], T]
) -> T:
    for topic_id in map(str, topic_ids):
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    return loader()


def document_fold(topic_id: str, document_id: str) -> int:
    if str(topic_id) in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")
    digest = hashlib.sha256(f"{topic_id}\0{document_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 2


def render_obligation_query(narrative: str, obligation: str) -> str:
    narrative, obligation = narrative.strip(), obligation.strip()
    if not narrative or not obligation:
        raise ValueError("narrative and obligation must be nonempty")
    return f"{narrative}\n\nExplicit obligation:\n{obligation}"


def _preflight_manifest_topic_ids(manifest: Mapping[str, object]) -> list[str]:
    declared = manifest.get("topic_ids")
    if declared is not None:
        if not isinstance(declared, list):
            raise ValueError("manifest topic_ids must be a JSON array")
        reject_protected_before_access((str(value) for value in declared), lambda: None)

    topic_rows = manifest.get("topics")
    if not isinstance(topic_rows, list):
        raise ValueError("manifest topics must be a JSON array")
    topic_ids = [str(row["topic_id"]) for row in topic_rows]
    reject_protected_before_access(topic_ids, lambda: None)

    facet_rows = manifest.get("facets")
    if not isinstance(facet_rows, list):
        raise ValueError("manifest facets must be a JSON array")
    facet_topic_ids = [str(row["topic_id"]) for row in facet_rows]
    reject_protected_before_access(facet_topic_ids, lambda: None)
    return topic_ids


def build_contract(
    manifest: Mapping[str, object],
    gates: Mapping[str, object],
    union_rows: Sequence[Mapping[str, object]],
    *,
    expected_population: int = 8114,
    expected_o0: int = 24,
) -> dict[str, object]:
    _preflight_manifest_topic_ids(manifest)
    topics = {str(row["topic_id"]): row for row in manifest["topics"]}  # type: ignore[index]
    accepted = {
        str(row["facet_id"])
        for row in gates["gates"]  # type: ignore[index]
        if row.get("accepted") is True
    }
    if len(accepted) != expected_o0:
        raise ValueError(
            f"accepted facet set must contain exactly {expected_o0} O0 obligations"
        )
    obligations: list[dict[str, object]] = []
    for topic_id in topics:
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        narrative = str(topics[topic_id]["query"])
        obligations.append(
            {
                "topic_id": topic_id,
                "obligation_id": f"{topic_id}:broad",
                "kind": "broad",
                "parent_id": None,
                "manifest_order": -1,
                "text": narrative,
                "query": narrative,
            }
        )
    materialized_accepted: list[str] = []
    for facet in manifest["facets"]:  # type: ignore[index]
        topic_id = str(facet["topic_id"])
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in topics:
            raise ValueError(f"facet topic {topic_id} is outside the manifest topic set")
        facet_id = str(facet["facet_id"])
        if facet_id not in accepted:
            continue
        materialized_accepted.append(facet_id)
        text = str(facet["obligation"])
        obligations.append(
            {
                "topic_id": topic_id,
                "obligation_id": facet_id,
                "source_facet_id": facet_id,
                "kind": "o0",
                "parent_id": None,
                "manifest_order": int(facet["manifest_order"]),
                "text": text,
                "query": render_obligation_query(str(topics[topic_id]["query"]), text),
                "anchor_terms": list(facet["anchor_terms"]),
                "relation_terms": list(facet["relation_terms"]),
                "wrong_domain_patterns": list(facet["wrong_domain_patterns"]),
            }
        )
    if len(materialized_accepted) != expected_o0 or set(materialized_accepted) != accepted:
        raise ValueError("accepted facet set does not match manifest facets")
    documents = []
    for row in union_rows:
        topic_id, document_id = str(row["topic_id"]), str(row["document_id"])
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in topics:
            raise ValueError(
                f"accepted-union topic {topic_id} is outside the manifest topic set"
            )
        text = str(row["text"])
        if hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("accepted-union text hash mismatch")
        documents.append({**dict(row), "fold": document_fold(topic_id, document_id)})
    if len(documents) != expected_population:
        raise ValueError(
            f"accepted population must contain exactly {expected_population} rows"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "obligations": obligations,
        "documents": documents,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _load_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, raw


def _load_manifest_topic_metadata(path: Path) -> tuple[list[str], list[str]]:
    payload, _ = _load_json(path, "manifest topic metadata")
    topics = payload.get("topics")
    if not isinstance(topics, list):
        raise ValueError("manifest topics must be a JSON array")
    record_topic_ids = [str(row["topic_id"]) for row in topics]

    declared = payload.get("topic_ids")
    if declared is None:
        declared_topic_ids = list(record_topic_ids)
    elif isinstance(declared, list):
        declared_topic_ids = [str(topic_id) for topic_id in declared]
    else:
        raise ValueError("manifest topic_ids must be a JSON array")
    return declared_topic_ids, record_topic_ids


def _manifest_topic_ids(manifest: Mapping[str, object]) -> list[str]:
    declared = manifest.get("topic_ids")
    if isinstance(declared, list):
        topic_ids = [str(topic_id) for topic_id in declared]
    else:
        topics = manifest.get("topics")
        if not isinstance(topics, list):
            raise ValueError("manifest topics must be a JSON array")
        topic_ids = [str(row["topic_id"]) for row in topics]

    reject_protected_before_access(topic_ids, lambda: None)
    if len(topic_ids) != len(PILOT_TOPIC_IDS) or set(topic_ids) != set(PILOT_TOPIC_IDS):
        raise ValueError(
            "manifest topic set must contain exactly the four pilot topics "
            f"{list(PILOT_TOPIC_IDS)}"
        )

    topics = manifest.get("topics")
    if not isinstance(topics, list):
        raise ValueError("manifest topics must be a JSON array")
    record_topic_ids = [str(row["topic_id"]) for row in topics]
    reject_protected_before_access(record_topic_ids, lambda: None)
    if len(record_topic_ids) != len(PILOT_TOPIC_IDS) or set(record_topic_ids) != set(
        PILOT_TOPIC_IDS
    ):
        raise ValueError("manifest topic records do not match the four pilot topics")

    facets = manifest.get("facets")
    if not isinstance(facets, list):
        raise ValueError("manifest facets must be a JSON array")
    facet_topic_ids = [str(row["topic_id"]) for row in facets]
    reject_protected_before_access(facet_topic_ids, lambda: None)
    if not set(facet_topic_ids).issubset(PILOT_TOPIC_IDS):
        raise ValueError("manifest facets contain a non-pilot topic")
    return topic_ids


def _load_cli_sources(
    *,
    manifest_topic_ids_loader: Callable[[], tuple[Sequence[str], Sequence[str]]],
    manifest_loader: Callable[[], tuple[dict[str, object], bytes]],
    gates_loader: Callable[[], tuple[dict[str, object], bytes]],
    union_loader: Callable[[], tuple[list[dict[str, object]], bytes]],
) -> tuple[
    dict[str, object],
    bytes,
    dict[str, object],
    bytes,
    list[dict[str, object]],
    bytes,
]:
    declared_topic_ids, record_topic_ids = manifest_topic_ids_loader()
    for topic_ids in (declared_topic_ids, record_topic_ids):
        reject_protected_before_access(topic_ids, lambda: None)
        if len(topic_ids) != len(PILOT_TOPIC_IDS) or set(topic_ids) != set(
            PILOT_TOPIC_IDS
        ):
            raise ValueError(
                "manifest topic set must contain exactly the four pilot topics "
                f"{list(PILOT_TOPIC_IDS)}"
            )

    manifest, manifest_raw = manifest_loader()
    _manifest_topic_ids(manifest)

    gates, gates_raw = gates_loader()
    gate_rows = gates.get("gates")
    if not isinstance(gate_rows, list):
        raise ValueError("gate records must be a JSON array")
    gate_topic_ids = [
        str(row["topic_id"]) for row in gate_rows if "topic_id" in row
    ]
    reject_protected_before_access(gate_topic_ids, lambda: None)
    if not set(gate_topic_ids).issubset(PILOT_TOPIC_IDS):
        raise ValueError("gate records contain a non-pilot topic")

    union_rows, union_raw = union_loader()
    return manifest, manifest_raw, gates, gates_raw, union_rows, union_raw


def _load_union(path: Path) -> tuple[list[dict[str, object]], bytes]:
    raw = path.read_bytes()
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"accepted union contains a blank line at {line_number}")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid accepted-union JSON at line {line_number}: {path}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"accepted-union line {line_number} must be a JSON object"
            )
        topic_id = str(row.get("topic_id"))
        reject_protected_before_access([topic_id], lambda: None)
        if topic_id not in PILOT_TOPIC_IDS:
            raise ValueError(
                f"accepted-union line {line_number} contains non-pilot topic {topic_id}"
            )
        rows.append(row)
    return rows, raw


def _create_contract_files(
    contract: Mapping[str, object],
    output: Path,
    *,
    source_records: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    obligations = contract["obligations"]
    documents = contract["documents"]
    if not isinstance(obligations, list) or not isinstance(documents, list):
        raise ValueError("contract obligations and documents must be arrays")

    topic_ids = [
        str(row["topic_id"])
        for row in obligations
        if row.get("kind") == "broad"
    ]
    reject_protected_before_access(topic_ids, lambda: None)
    if len(topic_ids) != len(PILOT_TOPIC_IDS) or set(topic_ids) != set(PILOT_TOPIC_IDS):
        raise ValueError("contract topic set does not match the four pilot topics")

    o0_count = sum(row.get("kind") == "o0" for row in obligations)
    broad_count = sum(row.get("kind") == "broad" for row in obligations)
    folds = [
        {
            "topic_id": str(row["topic_id"]),
            "document_id": str(row["document_id"]),
            "union_order": row["union_order"],
            "fold": row["fold"],
        }
        for row in documents
    ]
    manifest = {
        "schema_version": contract["schema_version"],
        "topic_ids": topic_ids,
        "protected_topic_ids": sorted(PROTECTED_TOPIC_IDS, key=int),
        "document_count": len(documents),
        "broad_obligation_count": broad_count,
        "o0_obligation_count": o0_count,
        "fold_assignment": "sha256(topic_id + NUL + document_id) mod 2",
        "sources": {name: dict(record) for name, record in source_records.items()},
        "qrels_opened": False,
    }
    artifact_bytes = {
        "manifest.json": _canonical_json_bytes(manifest),
        "obligations.jsonl": _canonical_jsonl_bytes(obligations),
        "documents.jsonl": _canonical_jsonl_bytes(documents),
        "folds.jsonl": _canonical_jsonl_bytes(folds),
    }
    artifact_sha256 = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in artifact_bytes.items()
    }
    fold_counts = {
        str(fold): sum(row["fold"] == fold for row in documents) for fold in (0, 1)
    }
    summary: dict[str, object] = {
        "schema_version": contract["schema_version"],
        "status": "complete",
        "topic_ids": topic_ids,
        "document_count": len(documents),
        "broad_obligation_count": broad_count,
        "o0_obligation_count": o0_count,
        "fold_counts": fold_counts,
        "protected_topic_count": 0,
        "qrels_opened": False,
        "artifact_sha256": artifact_sha256,
    }

    output.mkdir(parents=True, exist_ok=False)
    for name, content in artifact_bytes.items():
        with (output / name).open("xb") as handle:
            handle.write(content)
    with (output / "summary.json").open("xb") as handle:
        handle.write(_canonical_json_bytes(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="freeze the full-union contract")
    create.add_argument("--manifest", required=True, type=Path)
    create.add_argument("--gates", required=True, type=Path)
    create.add_argument("--union", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command != "create":  # pragma: no cover - argparse enforces this.
        parser.error("unsupported command")
    if args.output.exists():
        raise FileExistsError(f"contract output already exists: {args.output}")

    (
        manifest,
        manifest_raw,
        gates,
        gates_raw,
        union_rows,
        union_raw,
    ) = _load_cli_sources(
        manifest_topic_ids_loader=lambda: _load_manifest_topic_metadata(args.manifest),
        manifest_loader=lambda: _load_json(args.manifest, "source manifest"),
        gates_loader=lambda: _load_json(args.gates, "gate"),
        union_loader=lambda: _load_union(args.union),
    )
    contract = build_contract(manifest, gates, union_rows)
    summary = _create_contract_files(
        contract,
        args.output,
        source_records={
            "manifest": {
                "path": str(args.manifest),
                "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            },
            "gates": {
                "path": str(args.gates),
                "sha256": hashlib.sha256(gates_raw).hexdigest(),
            },
            "accepted_union": {
                "path": str(args.union),
                "sha256": hashlib.sha256(union_raw).hexdigest(),
            },
        },
    )
    print(
        "status=complete "
        f"documents={summary['document_count']} "
        f"o0={summary['o0_obligation_count']} "
        f"protected={summary['protected_topic_count']} "
        "qrels_opened=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
