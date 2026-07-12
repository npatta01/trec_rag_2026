"""Create-only offline freezer for facet-control ranking alternatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from .det_sparse_ledger import RetrievalLedger
from .facet_retrieval_control_experiment import (
    FACET_FAMILY_WEIGHT,
    ORIGINAL_FAMILY_WEIGHT,
    RANKING_DEPTH,
    RRF_K,
    build_topic_alternatives,
    index_control_streams,
)
from .facet_retrieval_control_inspector import inspect_stream, load_inspection_streams
from .facet_retrieval_control_manifest import (
    PROTECTED_TOPIC_IDS,
    ControlManifest,
    load_control_manifest,
)
from .facet_retrieval_control_run import build_control_requests
from .pipeline_models import RankedCandidate, RetrievedCandidate, jsonable


FREEZE_SCHEMA_VERSION = "facet-control-ranking-freeze-v1"


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_write(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _ranking_bytes(rows: Sequence[RankedCandidate]) -> bytes:
    return b"".join(
        json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _ranking_sha256(rows: Sequence[RankedCandidate]) -> str:
    payload = [jsonable(row) for row in rows]
    canonical = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_sha256(value: object, label: str) -> None:
    if isinstance(value, str):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{label} must be a lowercase SHA-256")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty text")
            _validate_sha256(nested, f"{label}.{key}")
        return
    raise ValueError(f"{label} must contain SHA-256 values")


def _validate_bindings(bindings: Mapping[str, object]) -> dict[str, object]:
    required = {
        "manifest_sha256",
        "prior_freeze_sha256",
        "request_sha256",
        "response_sha256",
        "candidate_sha256",
    }
    allowed = required | {"prior_ledger_sha256", "r1_arm_sha256"}
    if not required <= set(bindings) or not set(bindings) <= allowed:
        raise ValueError(
            f"freeze bindings must include exactly the required hashes {sorted(required)!r}"
        )
    result = dict(bindings)
    for key, value in result.items():
        _validate_sha256(value, key)
    return result


def create_control_freeze(
    output_dir: Path,
    rankings: Mapping[str, Sequence[RankedCandidate]],
    *,
    inspections: Mapping[str, object],
    bindings: Mapping[str, object],
) -> dict[str, object]:
    """Write all ranking evidence once and commit ``freeze.json`` with O_EXCL."""

    if len(rankings) != 25:
        raise ValueError("control freeze requires exactly 25 ranking alternatives")
    expected_counts = {
        "200": 4,
        "225": 16,
        "707": 4,
        "897": 1,
    }
    for topic_id, expected in expected_counts.items():
        observed = sum(key.startswith(f"R2:{topic_id}:") for key in rankings)
        if observed != expected:
            raise ValueError(
                f"control freeze requires {expected} alternatives for topic {topic_id}"
            )
    for key, rows in rankings.items():
        topic_id = key.split(":", 2)[1]
        if len(rows) != RANKING_DEPTH or any(row.topic_id != topic_id for row in rows):
            raise ValueError(f"ranking {key} must contain exactly 100 target-topic rows")
        if [row.rank for row in rows] != list(range(1, RANKING_DEPTH + 1)):
            raise ValueError(f"ranking {key} has non-canonical final ranks")

    frozen_bindings = _validate_bindings(bindings)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rankings_dir = output / "rankings"
    rankings_dir.mkdir()

    ranking_manifest: dict[str, dict[str, object]] = {}
    for key in sorted(rankings):
        filename = key.replace(":", "__") + ".jsonl"
        content = _ranking_bytes(rankings[key])
        _exclusive_write(rankings_dir / filename, content)
        ranking_manifest[key] = {
            "path": f"rankings/{filename}",
            "rows": len(rankings[key]),
            "sha256": _ranking_sha256(rankings[key]),
            "file_sha256": hashlib.sha256(content).hexdigest(),
        }

    inspection_bytes = _canonical_json(inspections)
    _exclusive_write(output / "inspection.json", inspection_bytes)
    fusion = {
        "k": RRF_K,
        "limit": RANKING_DEPTH,
        "original_family_weight": ORIGINAL_FAMILY_WEIGHT,
        "facet_family_weight": FACET_FAMILY_WEIGHT,
        "facet_stream_weight": "0.5 / active topic facet streams",
    }
    fusion_bytes = _canonical_json(fusion)
    _exclusive_write(output / "fusion.json", fusion_bytes)

    payload: dict[str, object] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen_before_qrels",
        "bindings": frozen_bindings,
        "inspection_sha256": hashlib.sha256(inspection_bytes).hexdigest(),
        "fusion_sha256": hashlib.sha256(fusion_bytes).hexdigest(),
        "rankings": ranking_manifest,
    }
    payload["freeze_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    _exclusive_write(output / "freeze.json", _canonical_json(payload))
    return payload


def verify_sha256(path: Path, expected_sha256: str) -> str:
    """Verify one input before any freeze output is created."""

    _validate_sha256(expected_sha256, str(path))
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"input SHA-256 mismatch for {path}")
    return actual


def verify_prior_freeze(path: Path) -> str:
    """Verify the prior qrels-blind freeze and every ranking it names."""

    freeze_path = Path(path)
    source = freeze_path.read_bytes()
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError("prior freeze is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "sparse-relevance-ranking-freeze-v1"
        or payload.get("status") != "frozen_before_qrels"
    ):
        raise ValueError("prior freeze has an invalid frozen-before-qrels contract")
    expected_self = payload.get("freeze_sha256")
    _validate_sha256(expected_self, "prior freeze_sha256")
    without_self = dict(payload)
    without_self.pop("freeze_sha256")
    actual_self = hashlib.sha256(
        json.dumps(
            without_self,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()
    if actual_self != expected_self:
        raise ValueError("prior freeze self SHA-256 is invalid")
    rankings = payload.get("rankings")
    if not isinstance(rankings, dict) or "R1:family_rrf" not in rankings:
        raise ValueError("prior freeze lacks the required R1 family ranking")
    for key, raw_record in rankings.items():
        if not isinstance(key, str) or not isinstance(raw_record, dict):
            raise ValueError("prior freeze ranking records are invalid")
        expected = raw_record.get("sha256")
        _validate_sha256(expected, f"prior ranking {key}")
        relative = raw_record.get("path")
        ranking_path = (
            freeze_path.parent / relative
            if isinstance(relative, str)
            else freeze_path.parent / "rankings" / f"{key.replace(':', '__')}.jsonl"
        )
        try:
            decoded_rows = [
                json.loads(line) for line in ranking_path.read_text(encoding="utf-8").splitlines() if line
            ]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"prior ranking is unreadable for {key}") from exc
        actual = hashlib.sha256(
            (
                json.dumps(
                    decoded_rows,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"prior ranking SHA-256 mismatch for {key}")
        rows = raw_record.get("rows")
        if not isinstance(rows, int) or rows < 1:
            raise ValueError(f"prior ranking row count is invalid for {key}")
        if len(decoded_rows) != rows:
            raise ValueError(f"prior ranking row count mismatch for {key}")
    return hashlib.sha256(source).hexdigest()


def _load_candidates_jsonl(path: Path) -> list[RetrievedCandidate]:
    rows: list[RetrievedCandidate] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            raw = json.loads(line)
            rows.append(RetrievedCandidate(**raw))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"invalid candidate JSONL row {line_number} in {path}") from exc
    if not rows:
        raise ValueError(f"candidate JSONL is empty: {path}")
    return rows


def _ledger_from_existing(run_dir: Path, shared_cache: Path | None) -> RetrievalLedger:
    policy_path = Path(run_dir) / "ledger.json"
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load ledger policy: {policy_path}") from exc
    if not isinstance(policy, dict):
        raise ValueError(f"ledger policy must be an object: {policy_path}")
    ledger = RetrievalLedger(
        run_dir,
        shared_cache_dir=shared_cache,
        max_calls=policy.get("max_calls"),
        max_calls_per_topic=policy.get("max_calls_per_topic"),
        min_results=policy.get("min_results"),
        required_text_results=policy.get("required_text_results"),
    )
    report = ledger.validate_run()
    if report.failures or report.pending:
        raise ValueError(f"ledger is not complete and successful: {run_dir}")
    return ledger


def _ledger_tree_sha256(run_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item for item in Path(run_dir).rglob("*") if item.is_file() and item.name != ".ledger.lock"
    ):
        digest.update(path.relative_to(run_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_control_candidates(
    manifest: ControlManifest,
    ledger_dir: Path,
    shared_cache: Path,
    endpoint: str,
) -> tuple[list[RetrievedCandidate], dict[str, str], dict[str, str], dict[str, str]]:
    ledger = _ledger_from_existing(ledger_dir, shared_cache)
    requests = build_control_requests(manifest, endpoint=endpoint)
    if ledger.validate_run().planned_requests != len(requests):
        raise ValueError("control ledger does not contain exactly the 12 frozen requests")
    rows: list[RetrievedCandidate] = []
    request_hashes: dict[str, str] = {}
    response_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    for request in requests:
        result = ledger.load_verified_result(request)
        key = request.identity.request_key
        request_hashes[key] = key
        response_hashes[key] = result.response_sha256
        candidate_hashes[key] = result.candidates_sha256
        rows.extend(
            RetrievedCandidate(
                topic_id=request.identity.topic_id,
                variant_name=request.identity.variant_name,
                retriever_name=request.identity.retriever_version,
                query_text=request.query_text,
                docid=candidate.docid,
                rank=candidate.rank,
                score=candidate.score,
                text=candidate.text,
            )
            for candidate in result.candidates
        )
    return rows, request_hashes, response_hashes, candidate_hashes


def freeze_control_experiment(
    *,
    output_dir: Path,
    r1_source_manifest: Path,
    r1_arm: Sequence[RetrievedCandidate],
    control_rows: Sequence[RetrievedCandidate],
    manifest: ControlManifest,
    bindings: Mapping[str, object],
) -> dict[str, object]:
    """Inspect all 16 stream arms and create all 25 alternatives without qrels."""

    indexed = index_control_streams(r1_arm, control_rows, manifest)
    specs = load_inspection_streams(r1_source_manifest, manifest)
    inspections: dict[str, object] = {}
    for stream in manifest.streams:
        for arm in ("B0", "W0", "W1", "W2"):
            key = f"{stream.topic_id}/{stream.stream_id}/{arm}"
            inspections[key] = jsonable(
                inspect_stream(
                    specs[(stream.topic_id, stream.stream_id)],
                    indexed[(stream.topic_id, stream.stream_id, arm)],
                )
            )
    rankings = build_topic_alternatives(r1_arm, control_rows, manifest)
    return create_control_freeze(
        output_dir,
        rankings,
        inspections=inspections,
        bindings=bindings,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--r1-source-manifest", type=Path, required=True)
    parser.add_argument("--prior-freeze", type=Path, required=True)
    parser.add_argument("--prior-ledger", type=Path, action="append", required=True)
    parser.add_argument("--r1-candidates", type=Path, required=True)
    parser.add_argument("--r1-candidates-sha256", required=True)
    parser.add_argument("--control-ledger", type=Path, required=True)
    parser.add_argument("--shared-cache", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_control_manifest(args.manifest)
    # The protected namespace gate precedes prior-freeze, ledger, cache, and fusion access.
    protected = sorted(
        {stream.topic_id for stream in manifest.streams} & set(PROTECTED_TOPIC_IDS)
    )
    if protected:
        raise ValueError(f"protected topic {protected[0]} is forbidden")
    if args.output.exists():
        raise FileExistsError(f"create-only output already exists: {args.output}")

    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    prior_freeze_sha256 = verify_prior_freeze(args.prior_freeze)
    verify_sha256(args.r1_candidates, args.r1_candidates_sha256)
    r1_arm = _load_candidates_jsonl(args.r1_candidates)
    candidate_protected = sorted(
        {row.topic_id for row in r1_arm} & set(PROTECTED_TOPIC_IDS)
    )
    if candidate_protected:
        raise ValueError(f"protected topic {candidate_protected[0]} is forbidden")
    prior_ledgers: dict[str, str] = {}
    for path in sorted(args.prior_ledger):
        _ledger_from_existing(path, args.shared_cache)
        prior_ledgers[str(path)] = _ledger_tree_sha256(path)
    control_rows, request_hashes, response_hashes, candidate_hashes = (
        _load_control_candidates(
            manifest,
            args.control_ledger,
            args.shared_cache,
            args.endpoint,
        )
    )
    bindings: dict[str, object] = {
        "manifest_sha256": manifest_sha256,
        "prior_freeze_sha256": prior_freeze_sha256,
        "prior_ledger_sha256": prior_ledgers,
        "r1_arm_sha256": args.r1_candidates_sha256,
        "request_sha256": request_hashes,
        "response_sha256": response_hashes,
        "candidate_sha256": {
            "r1_arm": args.r1_candidates_sha256,
            **candidate_hashes,
        },
    }
    freeze = freeze_control_experiment(
        output_dir=args.output,
        r1_source_manifest=args.r1_source_manifest,
        r1_arm=r1_arm,
        control_rows=control_rows,
        manifest=manifest,
        bindings=bindings,
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
