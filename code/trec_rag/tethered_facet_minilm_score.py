"""Authenticated, tokenizer-only preflight for narrative-tethered facet scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from .facet_local_minilm_preflight import (
    MODEL_ID,
    MODEL_REVISION,
    PAIR_MAX_TOKENS,
    build_window_plan,
    load_verified_materialization,
    load_verified_tokenizer,
    score_cache_context,
)
from .facet_local_minilm_rank import aggregate_top4
from .rerank_score_cache import GlobalScoreCache


TOPIC_IDS = ("219", "72", "300", "84")
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
PAIR_COUNT = 4_800
PHASE1_PAIR_COUNT = 5_000
ACCEPTED_FACET_COUNT = 24
ACCEPTED_UNION_COUNTS = {"219": 2_182, "72": 2_127, "300": 1_712, "84": 2_093}
ACCEPTED_UNION_COUNT = 8_114
WINDOW_CEILING = 25_000
RUNTIME_CEILING_SECONDS = 600.0
REFERENCE_FIXED_SECONDS = 30.0
PUBLICATION_SUFFIX = ("post_qrels_tethered_facet_minilm_v1", "scoring")
SCORE_CACHE_ROOT = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/reranker"
)
PREFLIGHT_SCHEMA_VERSION = "tethered-facet-minilm-preflight-v1"
CANDIDATE_SCHEMA_VERSION = "tethered-facet-minilm-candidate-v1"
SCORE_SCHEMA_VERSION = "tethered-facet-minilm-score-v1"
DOCUMENT_SCORE_SCHEMA_VERSION = "tethered-facet-minilm-document-score-v1"
SCORING_RECEIPT_SCHEMA_VERSION = "tethered-facet-minilm-scoring-receipt-v1"
BATCH_SIZE = 32

_SCORING_RECEIPT_REQUIRED = frozenset(
    {
        "schema_version",
        "status",
        "phase",
        "qrels_opened",
        "network_access_supported",
        "hosted_inference_supported",
        "model",
        "model_revision",
        "preflight_sha256",
        "windows_sha256",
        "scores_sha256",
        "planned_window_count",
        "completed_window_count",
        "unique_forward_pair_count",
        "cache_reuse_pair_count",
        "elapsed_seconds",
        "peak_device_memory_bytes",
        "peak_host_memory_bytes",
        "device",
        "execution_backend",
    }
)
_PHASE1_PREFLIGHT_REQUIRED = frozenset(
    {
        "schema_version",
        "status",
        "phase",
        "qrels_opened",
        "network_access_supported",
        "hosted_inference_supported",
        "model_constructed",
        "model",
        "model_revision",
        "candidates_sha256",
        "windows_sha256",
        "model_materialization_receipt",
        "model_materialization_receipt_sha256",
        "summary",
        "pairs_per_second",
        "fixed_seconds",
        "projected_runtime_seconds",
        "runtime_ceiling_seconds",
    }
)
_PHASE1_SUMMARY_REQUIRED = frozenset(
    {
        "document_count",
        "window_count",
        "unique_pair_count",
        "unique_uncached_pair_count",
        "cache_hit_window_count",
    }
)


class _TokenizerOnlyBackend:
    """Small adapter exposing only the tokenization operations used by planning."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> list[int]:
        if add_special_tokens or truncation:
            raise ValueError("tokenizer-only planning requires raw, untruncated tokens")
        return list(
            self._tokenizer.encode(  # type: ignore[attr-defined]
                text, add_special_tokens=False
            ).ids
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise ValueError("tokenizer-only planning requires exact token decoding")
        return str(
            self._tokenizer.decode(  # type: ignore[attr-defined]
                list(token_ids), skip_special_tokens=False
            )
        )

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        return int(
            self._tokenizer.post_processor.num_special_tokens_to_add(  # type: ignore[attr-defined]
                pair
            )
        )


class TokenizerOnlyAuto:
    """`AutoTokenizer`-shaped local loader that never imports a model runtime."""

    @classmethod
    def from_pretrained(
        cls,
        snapshot: Path,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
        use_fast: bool,
    ) -> _TokenizerOnlyBackend:
        if not local_files_only or trust_remote_code or not use_fast:
            raise ValueError("tokenizer-only loader requires the frozen safe settings")
        from tokenizers import Tokenizer

        return _TokenizerOnlyBackend(Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json")))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _compact_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_compact_bytes(row) + b"\n" for row in rows)


def _read_object(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        source = Path(path).read_bytes()
        value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, source


def _require_fields(
    value: Mapping[str, object], required: frozenset[str], label: str
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} required fields missing: {', '.join(missing)}")


def _required_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} required SHA-256 is invalid")
    return value


def _required_positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} required count is invalid")
    return value


def _required_positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} required value is invalid")
    result = float(value)
    if not (result > 0.0 and result < float("inf")):
        raise ValueError(f"{label} required value is invalid")
    return result


def _require_publication_path(path: Path) -> Path:
    destination = Path(path)
    if tuple(destination.parts[-2:]) != PUBLICATION_SUFFIX:
        raise ValueError(
            "preflight output must end exactly with "
            "post_qrels_tethered_facet_minilm_v1/scoring"
        )
    return destination


def probe_working_rocm_device() -> dict[str, object]:
    """Probe one real ROCm allocation in an isolated process, without a model."""

    helper = Path(sys.executable).with_name("python-rocm")
    executable = helper if helper.is_file() else Path(sys.executable)
    script = """
import json
import torch
if not torch.cuda.is_available() or not torch.version.hip:
    raise SystemExit("working ROCm device is unavailable")
torch.cuda.reset_peak_memory_stats()
probe = torch.zeros(1, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
payload = {
    "available": True,
    "execution_backend": "rocm",
    "device": "cuda",
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "hip_version": str(torch.version.hip),
    "probe_allocation_bytes": probe.nelement() * probe.element_size(),
    "peak_device_memory_bytes": torch.cuda.max_memory_allocated(),
}
print(json.dumps(payload, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [str(executable), "-c", script],
            check=True,
            capture_output=True,
            env={**os.environ, "TMPDIR": "/var/tmp"},
            text=True,
            timeout=30,
        )
        value = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ValueError("working ROCm device is required") from exc
    if not isinstance(value, dict):
        raise ValueError("working ROCm device probe is invalid")
    return value


def _validate_device_probe(value: Mapping[str, object]) -> dict[str, object]:
    required = frozenset(
        {
            "available",
            "execution_backend",
            "device",
            "device_count",
            "device_name",
            "hip_version",
            "probe_allocation_bytes",
            "peak_device_memory_bytes",
        }
    )
    _require_fields(value, required, "ROCm device probe")
    if (
        value.get("available") is not True
        or value.get("execution_backend") != "rocm"
        or value.get("device") != "cuda"
        or not isinstance(value.get("device_name"), str)
        or not str(value.get("device_name")).strip()
        or not isinstance(value.get("hip_version"), str)
        or not str(value.get("hip_version")).strip()
    ):
        raise ValueError("working ROCm device is required")
    try:
        _required_positive_int(value.get("device_count"), "ROCm device count")
    except ValueError as exc:
        raise ValueError("working ROCm device is required") from exc
    _required_positive_int(
        value.get("probe_allocation_bytes"), "ROCm probe allocation"
    )
    _required_positive_int(
        value.get("peak_device_memory_bytes"), "ROCm peak device memory"
    )
    return dict(value)


def _read_jsonl_source(source: bytes, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = source.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    for number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{number} must be an object")
        rows.append(value)
    return rows


def _exclusive_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            sink.write(value)
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        os.close(descriptor)


def render_tethered_query(narrative: str, facet_query: str) -> str:
    narrative = narrative.strip()
    facet_query = facet_query.strip()
    if not narrative:
        raise ValueError("narrative is required")
    if not facet_query:
        raise ValueError("facet query is required")
    return f"{narrative}\n\nFocus: {facet_query}"


def reject_topic(value: object) -> str:
    topic_id = str(value)
    if topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")
    if topic_id not in TOPIC_IDS:
        raise ValueError(f"unexpected topic {topic_id}")
    return topic_id


def _rows(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{label} rows must be objects")
    return list(value)  # type: ignore[arg-type]


def _required_text(row: Mapping[str, object], key: str, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} {key} must be non-empty text")
    return value


def build_tethered_candidates(
    manifest: Mapping[str, object],
    phase1_rows: Sequence[Mapping[str, object]],
    gate_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join accepted facets to their phase-1 candidates without losing identity."""

    topic_ids = manifest.get("topic_ids")
    if topic_ids != list(TOPIC_IDS):
        raise ValueError("manifest topic IDs differ from the frozen order")
    topics = _rows(manifest.get("topics"), "manifest topics")
    facets = _rows(manifest.get("facets"), "manifest facets")

    narratives: dict[str, str] = {}
    for topic in topics:
        topic_id = reject_topic(topic.get("topic_id"))
        if topic_id in narratives:
            raise ValueError("manifest has duplicate topic IDs")
        narratives[topic_id] = _required_text(topic, "query", "manifest topic")
    if set(narratives) != set(TOPIC_IDS):
        raise ValueError("manifest topics do not cover the frozen topic IDs")

    facet_by_id: dict[str, Mapping[str, object]] = {}
    for facet in facets:
        topic_id = reject_topic(facet.get("topic_id"))
        facet_id = _required_text(facet, "facet_id", "manifest facet")
        facet_query = _required_text(facet, "query", "manifest facet")
        order = facet.get("manifest_order")
        if type(order) is not int or order < 0:
            raise ValueError("manifest facet order must be a non-negative integer")
        if facet_id in facet_by_id:
            raise ValueError("manifest has duplicate facet IDs")
        facet_by_id[facet_id] = {
            **facet,
            "topic_id": topic_id,
            "query": facet_query,
        }

    accepted: dict[str, Mapping[str, object]] = {}
    for gate in gate_rows:
        topic_id = reject_topic(gate.get("topic_id"))
        facet_id = _required_text(gate, "facet_id", "gate")
        if gate.get("status") != "accepted":
            continue
        facet = facet_by_id.get(facet_id)
        if facet is None or facet["topic_id"] != topic_id:
            raise ValueError("accepted gate is absent from the manifest")
        if gate.get("manifest_order") != facet.get("manifest_order"):
            raise ValueError("accepted gate manifest order differs")
        if facet_id in accepted:
            raise ValueError("duplicate accepted gate")
        accepted[facet_id] = facet

    output: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for source in phase1_rows:
        topic_id = reject_topic(source.get("topic_id"))
        facet_id = _required_text(source, "facet_id", "phase-1 candidate")
        facet = accepted.get(facet_id)
        if facet is None:
            continue
        if topic_id != facet["topic_id"]:
            raise ValueError("phase-1 candidate topic differs from facet")
        if source.get("manifest_order") != facet.get("manifest_order"):
            raise ValueError("phase-1 candidate manifest order differs")
        facet_query = str(facet["query"])
        if source.get("query") != facet_query:
            raise ValueError("phase-1 candidate query differs from manifest")
        source_query_hash = source.get("query_sha256")
        if source_query_hash is not None and source_query_hash != _sha256_text(
            facet_query
        ):
            raise ValueError("phase-1 candidate query hash differs")
        document_id = _required_text(
            source, "document_id", "phase-1 candidate"
        )
        if source.get("docid", document_id) != document_id:
            raise ValueError("phase-1 candidate document identities differ")
        identity = (facet_id, document_id)
        if identity in seen:
            raise ValueError("duplicate phase-1 facet/document identity")
        seen.add(identity)
        rank = source.get("rank")
        if type(rank) is not int or rank <= 0:
            raise ValueError("phase-1 candidate rank must be a positive integer")
        text = _required_text(source, "text", "phase-1 candidate")
        source_text_hash = source.get("text_sha256")
        if source_text_hash is not None and source_text_hash != _sha256_text(text):
            raise ValueError("phase-1 candidate text hash differs")
        query = render_tethered_query(narratives[topic_id], facet_query)
        output.append(
            {
                **source,
                "topic_id": topic_id,
                "family": "tethered_facet",
                "variant": facet_id,
                "facet_id": facet_id,
                "facet_query": facet_query,
                "facet_query_sha256": _sha256_text(facet_query),
                "query": query,
                "query_sha256": _sha256_text(query),
                "document_id": document_id,
                "docid": document_id,
                "text": text,
                "text_sha256": _sha256_text(text),
                "prior_bm25_rank": rank,
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )

    missing = set(accepted) - {str(row["facet_id"]) for row in output}
    if missing:
        raise ValueError("accepted facets lack phase-1 candidates")
    output.sort(
        key=lambda row: (
            TOPIC_IDS.index(str(row["topic_id"])),
            int(row["manifest_order"]),
            int(row["prior_bm25_rank"]),
            str(row["document_id"]),
        )
    )
    return output


def enforce_preflight_ceiling(
    pair_count: int, window_count: int, projected_seconds: float
) -> None:
    """Reject any drift from the frozen population or resource ceilings."""

    if (
        pair_count != PAIR_COUNT
        or window_count > WINDOW_CEILING
        or projected_seconds > RUNTIME_CEILING_SECONDS
    ):
        raise ValueError(
            "preflight ceiling violated: "
            f"pairs={pair_count}, windows={window_count}, seconds={projected_seconds}"
        )


def _source_path(
    sources: Mapping[str, object], key: str, *, child: str | None = None
) -> Path:
    value = sources.get(key)
    if not isinstance(value, (str, Path)):
        raise ValueError(f"preflight source {key} is required")
    path = Path(value)
    return path / child if child is not None else path


def _validate_manifest_hashes(manifest: Mapping[str, object]) -> None:
    _require_fields(
        manifest,
        frozenset(
            {
                "schema_version",
                "experiment_id",
                "topic_ids",
                "topics",
                "facets",
                "qrels_opened",
                "hashes",
            }
        ),
        "deep-facet manifest",
    )
    if (
        manifest.get("schema_version")
        != "rag25_deep_facet_candidate_manifest_v1"
        or manifest.get("experiment_id") != "rag25_deep_facet_candidates_v1"
    ):
        raise ValueError("deep-facet manifest identity differs")
    for topic in _rows(manifest.get("topics"), "manifest topics"):
        reject_topic(topic.get("topic_id"))
    for facet in _rows(manifest.get("facets"), "manifest facets"):
        reject_topic(facet.get("topic_id"))
    if manifest.get("qrels_opened") is not False:
        raise ValueError("manifest records qrels access")
    hashes = manifest.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("deep-facet manifest required hashes must be an object")
    _require_fields(
        hashes,
        frozenset({"topics_sha256", "facets_sha256", "freeze_sha256"}),
        "deep-facet manifest hashes",
    )
    expected_topics = _sha256_bytes(_compact_bytes(manifest.get("topics")) + b"\n")
    expected_facets = _sha256_bytes(_compact_bytes(manifest.get("facets")) + b"\n")
    unhashed = {key: value for key, value in manifest.items() if key != "hashes"}
    expected_freeze = _sha256_bytes(_compact_bytes(unhashed) + b"\n")
    if (
        hashes.get("topics_sha256") != expected_topics
        or hashes.get("facets_sha256") != expected_facets
        or hashes.get("freeze_sha256") != expected_freeze
    ):
        raise ValueError("manifest hashes differ from canonical content")


def _cache_binding(cache: object) -> dict[str, object]:
    value = getattr(cache, "path", None)
    if not isinstance(value, (str, Path)):
        raise ValueError("score cache must expose its bound path")
    path = Path(value).resolve()
    if not path.exists():
        return {"state": "absent", "path": str(path), "bytes": 0, "sha256": None}
    if not path.is_file():
        raise ValueError("score cache path must be a regular file")
    source = path.read_bytes()
    return {
        "state": "present",
        "path": str(path),
        "bytes": len(source),
        "sha256": _sha256_bytes(source),
    }


def _load_authenticated_accepted_union(
    gate_dir: Path,
    gate_summary: Mapping[str, object],
) -> tuple[set[tuple[str, str]], dict[str, object]]:
    """Authenticate, then stream the accepted union without materializing rows."""

    topic_counts = gate_summary.get("topic_counts")
    if not isinstance(topic_counts, Mapping):
        raise ValueError("gate summary required topic_counts is invalid")
    if set(map(str, topic_counts)) != set(TOPIC_IDS):
        raise ValueError("accepted union topic counts differ from frozen topics")
    for topic_id in TOPIC_IDS:
        raw = topic_counts.get(topic_id)
        if not isinstance(raw, Mapping) or raw.get("accepted_union") != ACCEPTED_UNION_COUNTS[topic_id]:
            raise ValueError("accepted union topic counts differ from frozen values")
    if sum(ACCEPTED_UNION_COUNTS.values()) != ACCEPTED_UNION_COUNT:
        raise AssertionError("accepted union constants do not reconcile")

    artifacts = gate_summary.get("artifacts")
    binding = artifacts.get("u_accepted.jsonl") if isinstance(artifacts, Mapping) else None
    if not isinstance(binding, Mapping):
        raise ValueError("gate summary required accepted union binding is missing")
    _require_fields(
        binding,
        frozenset({"bytes", "sha256"}),
        "gate summary accepted union binding",
    )
    expected_bytes = _required_positive_int(
        binding.get("bytes"), "accepted union bytes"
    )
    expected_sha = _required_sha256(
        binding.get("sha256"), "accepted union"
    )
    path = Path(gate_dir) / "u_accepted.jsonl"
    digest = hashlib.sha256()
    actual_bytes = 0
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                actual_bytes += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("accepted union is unreadable") from exc
    if actual_bytes != expected_bytes or digest.hexdigest() != expected_sha:
        raise ValueError("accepted union bytes/hash differ from gate summary")

    identities: set[tuple[str, str]] = set()
    observed: Counter[str] = Counter()
    try:
        with path.open("rb") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"accepted union:{line_number} is invalid JSON"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"accepted union:{line_number} must be an object")
                topic_id = reject_topic(row.get("topic_id"))
                document_id = row.get("document_id")
                if (
                    row.get("schema_version") != "deep-facet-candidate-union-v1"
                    or row.get("union") != "accepted"
                    or not isinstance(document_id, str)
                    or not document_id
                ):
                    raise ValueError(f"accepted union:{line_number} identity is invalid")
                identity = (topic_id, document_id)
                if identity in identities:
                    raise ValueError("accepted union contains a duplicate identity")
                identities.add(identity)
                observed[topic_id] += 1
    except OSError as exc:
        raise ValueError("accepted union changed while streaming") from exc
    if len(identities) != ACCEPTED_UNION_COUNT or dict(observed) != ACCEPTED_UNION_COUNTS:
        raise ValueError("accepted union actual topic counts differ")
    return identities, {
        "path": str(path.resolve()),
        "bytes": actual_bytes,
        "sha256": expected_sha,
        "rows": len(identities),
        "topic_counts": dict(observed),
    }


def _materialization_path(
    sources: Mapping[str, object],
    phase1_preflight: Mapping[str, object],
) -> tuple[Path, str]:
    receipt_value = phase1_preflight.get("model_materialization_receipt")
    expected = _required_sha256(
        phase1_preflight.get("model_materialization_receipt_sha256"),
        "phase-1 model materialization receipt",
    )
    if not isinstance(receipt_value, str) or not receipt_value:
        raise ValueError("phase-1 required model materialization path is invalid")
    receipt_path = Path(receipt_value)
    explicit = sources.get("model_receipt")
    if isinstance(explicit, (str, Path)):
        if Path(explicit).resolve() != receipt_path.resolve():
            raise ValueError("explicit model receipt differs from phase-1 binding")
        receipt_path = Path(explicit)
    return receipt_path, expected


def create_preflight(
    sources: Mapping[str, object],
    output: Path,
    *,
    tokenizer: object | None = None,
    cache: object | None = None,
    device_probe: Callable[[], Mapping[str, object]] = probe_working_rocm_device,
) -> dict[str, object]:
    """Authenticate sealed inputs and persist a create-only tokenizer plan."""

    started = time.perf_counter()
    destination = _require_publication_path(Path(output))
    if destination.exists():
        raise FileExistsError(f"create-only preflight output exists: {destination}")

    manifest_path = _source_path(sources, "manifest")
    phase1_dir = _source_path(sources, "phase1")
    gate_dir = _source_path(sources, "gate")
    manifest, manifest_source = _read_object(manifest_path, "deep-facet manifest")
    if manifest_source != _pretty_bytes(manifest):
        raise ValueError("deep-facet manifest must use canonical JSON bytes")
    _validate_manifest_hashes(manifest)
    scoring_receipt_path = phase1_dir / "scoring_receipt.json"
    scoring_receipt, scoring_source = _read_object(
        scoring_receipt_path, "phase-1 scoring receipt"
    )
    _require_fields(
        scoring_receipt,
        _SCORING_RECEIPT_REQUIRED,
        "phase-1 scoring receipt",
    )
    if (
        scoring_receipt.get("schema_version")
        != "deep-facet-candidate-minilm-receipt-v1"
        or scoring_receipt.get("status") != "complete"
        or scoring_receipt.get("phase") != "phase1_facet_local"
        or scoring_receipt.get("qrels_opened") is not False
        or scoring_receipt.get("network_access_supported") is not False
        or scoring_receipt.get("hosted_inference_supported") is not False
        or scoring_receipt.get("model") != MODEL_ID
        or scoring_receipt.get("model_revision") != MODEL_REVISION
        or scoring_receipt.get("device") != "cuda"
        or scoring_receipt.get("execution_backend") != "rocm"
    ):
        raise ValueError("phase-1 scoring receipt identity differs")
    expected_preflight_sha = _required_sha256(
        scoring_receipt.get("preflight_sha256"), "phase-1 preflight receipt"
    )
    expected_windows_sha = _required_sha256(
        scoring_receipt.get("windows_sha256"), "phase-1 windows"
    )
    expected_scores_sha = _required_sha256(
        scoring_receipt.get("scores_sha256"), "phase-1 scores"
    )
    planned_windows = _required_positive_int(
        scoring_receipt.get("planned_window_count"), "phase-1 planned windows"
    )
    completed_windows = _required_positive_int(
        scoring_receipt.get("completed_window_count"), "phase-1 completed windows"
    )
    prior_forward_pairs = _required_positive_int(
        scoring_receipt.get("unique_forward_pair_count"),
        "phase-1 unique forward pairs",
    )
    _required_positive_int(
        scoring_receipt.get("cache_reuse_pair_count"),
        "phase-1 cache reuse pairs",
        allow_zero=True,
    )
    prior_elapsed = _required_positive_float(
        scoring_receipt.get("elapsed_seconds"), "phase-1 elapsed seconds"
    )
    prior_peak_device = _required_positive_int(
        scoring_receipt.get("peak_device_memory_bytes"),
        "phase-1 peak device memory",
    )
    prior_peak_host = _required_positive_int(
        scoring_receipt.get("peak_host_memory_bytes"),
        "phase-1 peak host memory",
    )
    if completed_windows != planned_windows:
        raise ValueError("phase-1 scoring receipt window counts differ")

    phase1_preflight_path = phase1_dir / "preflight.json"
    phase1_preflight, phase1_preflight_source = _read_object(
        phase1_preflight_path, "phase-1 preflight receipt"
    )
    if _sha256_bytes(phase1_preflight_source) != expected_preflight_sha:
        raise ValueError("phase-1 preflight receipt hash differs")
    _require_fields(
        phase1_preflight,
        _PHASE1_PREFLIGHT_REQUIRED,
        "phase-1 preflight receipt",
    )
    if (
        phase1_preflight.get("schema_version")
        != "deep-facet-candidate-minilm-preflight-v1"
        or phase1_preflight.get("status") != "tokenizer_only_preflight_complete"
        or phase1_preflight.get("phase") != "phase1_facet_local"
        or phase1_preflight.get("qrels_opened") is not False
        or phase1_preflight.get("network_access_supported") is not False
        or phase1_preflight.get("hosted_inference_supported") is not False
        or phase1_preflight.get("model_constructed") is not False
        or phase1_preflight.get("model") != MODEL_ID
        or phase1_preflight.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("phase-1 preflight receipt identity differs")
    if phase1_preflight.get("windows_sha256") != expected_windows_sha:
        raise ValueError("phase-1 preflight/scoring window hashes differ")
    raw_phase1_summary = phase1_preflight.get("summary")
    if not isinstance(raw_phase1_summary, Mapping):
        raise ValueError("phase-1 preflight required summary is invalid")
    _require_fields(
        raw_phase1_summary,
        _PHASE1_SUMMARY_REQUIRED,
        "phase-1 preflight summary",
    )
    if (
        raw_phase1_summary.get("document_count") != PHASE1_PAIR_COUNT
        or raw_phase1_summary.get("window_count") != planned_windows
        or raw_phase1_summary.get("unique_uncached_pair_count")
        != prior_forward_pairs
    ):
        raise ValueError("phase-1 preflight expected counts differ")
    prior_unique_pairs = _required_positive_int(
        raw_phase1_summary.get("unique_pair_count"),
        "phase-1 preflight unique pairs",
    )
    prior_cache_hit_windows = _required_positive_int(
        raw_phase1_summary.get("cache_hit_window_count"),
        "phase-1 preflight cache-hit windows",
        allow_zero=True,
    )
    prior_cache_reuse = int(scoring_receipt["cache_reuse_pair_count"])
    if (
        prior_unique_pairs > planned_windows
        or prior_forward_pairs > prior_unique_pairs
        or prior_cache_reuse != prior_unique_pairs - prior_forward_pairs
        or prior_cache_hit_windows > planned_windows
    ):
        raise ValueError("phase-1 preflight/scoring cache counts differ")
    prior_policy_rate = _required_positive_float(
        phase1_preflight.get("pairs_per_second"),
        "phase-1 preflight pairs per second",
    )
    prior_fixed_seconds = _required_positive_float(
        phase1_preflight.get("fixed_seconds"),
        "phase-1 preflight fixed seconds",
    )
    prior_projected_seconds = _required_positive_float(
        phase1_preflight.get("projected_runtime_seconds"),
        "phase-1 preflight projected runtime",
    )
    if (
        prior_fixed_seconds != REFERENCE_FIXED_SECONDS
        or phase1_preflight.get("runtime_ceiling_seconds")
        != RUNTIME_CEILING_SECONDS
        or prior_projected_seconds
        != prior_fixed_seconds + prior_forward_pairs / prior_policy_rate
    ):
        raise ValueError("phase-1 preflight required runtime policy differs")
    for filename, field in (
        ("windows.jsonl", "windows_sha256"),
        ("scores.jsonl", "scores_sha256"),
    ):
        expected = expected_windows_sha if field == "windows_sha256" else expected_scores_sha
        bound_source = (phase1_dir / filename).read_bytes()
        if _sha256_bytes(bound_source) != expected:
            raise ValueError(f"phase-1 {filename} receipt hash differs")
        if len(bound_source.splitlines()) != planned_windows:
            raise ValueError(f"phase-1 {filename} required row count differs")

    gate_summary_path = gate_dir / "summary.json"
    gates_path = gate_dir / "gates.json"
    gate_summary, gate_summary_source = _read_object(gate_summary_path, "gate summary")
    gates, gates_source = _read_object(gates_path, "gates")
    _require_fields(
        gate_summary,
        frozenset(
            {
                "schema_version",
                "status",
                "qrels_opened",
                "facet_count",
                "accepted_facet_count",
                "rejected_facet_count",
                "topic_counts",
                "artifacts",
            }
        ),
        "gate summary",
    )
    _require_fields(
        gates,
        frozenset({"schema_version", "gates"}),
        "gates receipt",
    )
    artifact = gate_summary.get("artifacts")
    gate_binding = artifact.get("gates.json") if isinstance(artifact, Mapping) else None
    if isinstance(gate_binding, Mapping):
        _require_fields(
            gate_binding,
            frozenset({"bytes", "sha256"}),
            "gate summary gates binding",
        )
    if (
        gate_summary.get("schema_version") != "deep-facet-candidate-gate-v1"
        or gates.get("schema_version") != "deep-facet-candidate-gate-v1"
        or gate_summary.get("status") != "complete"
        or gate_summary.get("qrels_opened") is not False
        or not isinstance(gate_binding, Mapping)
        or gate_binding.get("bytes") != len(gates_source)
        or gate_binding.get("sha256") != _sha256_bytes(gates_source)
    ):
        raise ValueError("gate summary does not authenticate gates.json")
    raw_gates = gates.get("gates")
    gate_rows = _rows(raw_gates, "gates")
    accepted_count = sum(row.get("status") == "accepted" for row in gate_rows)
    if (
        gate_summary.get("facet_count") != 25
        or len(gate_rows) != 25
        or gate_summary.get("accepted_facet_count") != ACCEPTED_FACET_COUNT
        or accepted_count != ACCEPTED_FACET_COUNT
        or gate_summary.get("rejected_facet_count") != 1
    ):
        raise ValueError("gate required exact facet counts differ")
    accepted_union, accepted_union_binding = _load_authenticated_accepted_union(
        gate_dir, gate_summary
    )

    model_receipt_path, expected_model_sha = _materialization_path(
        sources, phase1_preflight
    )
    materialization = load_verified_materialization(model_receipt_path)
    if materialization.sha256 != expected_model_sha:
        raise ValueError("model materialization receipt hash differs")
    if (
        materialization.payload.get("model_id") != MODEL_ID
        or materialization.payload.get("revision") != MODEL_REVISION
    ):
        raise ValueError("model materialization identity differs")
    device_evidence = _validate_device_probe(device_probe())

    # Candidate bytes are opened only after all controlling receipts are authenticated.
    candidates_path = phase1_dir / "candidates.jsonl"
    try:
        phase1_source = candidates_path.read_bytes()
    except OSError as exc:
        raise ValueError("phase-1 candidates are unreadable") from exc
    expected_candidates_sha = _required_sha256(
        phase1_preflight.get("candidates_sha256"), "phase-1 candidates"
    )
    if expected_candidates_sha != _sha256_bytes(phase1_source):
        raise ValueError("phase-1 candidates hash differs")
    phase1_rows = _read_jsonl_source(phase1_source, "phase-1 candidates")
    if len(phase1_rows) != PHASE1_PAIR_COUNT:
        raise ValueError("phase-1 candidates require exactly 5,000 rows")
    for row in phase1_rows:
        query = row.get("query")
        text = row.get("text")
        if (
            not isinstance(query, str)
            or not isinstance(text, str)
            or row.get("query_sha256") != _sha256_text(query)
            or row.get("text_sha256") != _sha256_text(text)
        ):
            raise ValueError("phase-1 candidate required identity hashes differ")
    candidates = build_tethered_candidates(manifest, phase1_rows, gate_rows)
    facet_counts = Counter(str(row["facet_id"]) for row in candidates)
    if (
        len(candidates) != PAIR_COUNT
        or len(facet_counts) != ACCEPTED_FACET_COUNT
        or set(facet_counts.values()) != {200}
    ):
        raise ValueError(
            "tethered preflight requires exactly 4,800 pairs in 24 accepted "
            "facet streams of 200"
        )
    missing_from_union = next(
        (
            (str(row["topic_id"]), str(row["document_id"]))
            for row in candidates
            if (str(row["topic_id"]), str(row["document_id"]))
            not in accepted_union
        ),
        None,
    )
    if missing_from_union is not None:
        raise ValueError(
            "accepted facet candidate is absent from accepted union: "
            f"{missing_from_union[0]}/{missing_from_union[1]}"
        )

    if tokenizer is None:
        tokenizer = load_verified_tokenizer(
            model_receipt_path, auto_tokenizer_cls=TokenizerOnlyAuto
        )
    if cache is None:
        cache_root = sources.get("cache_root", SCORE_CACHE_ROOT)
        cache = GlobalScoreCache(Path(cache_root), score_cache_context())
    cache_before = _cache_binding(cache)

    window_rows: list[dict[str, object]] = []
    hits = 0
    topic_windows: Counter[str] = Counter()
    facet_windows: Counter[str] = Counter()
    coverage: dict[str, list[float]] = {}
    unique_pair_keys: set[str] = set()
    unique_miss_keys: set[str] = set()
    for candidate in candidates:
        planned = build_window_plan(candidate, tokenizer, query=str(candidate["query"]))
        local_coverage: list[float] = []
        for row in planned:
            cached = cache.get(query_text=row.query, text=row.window_text) is not None  # type: ignore[attr-defined]
            hits += int(cached)
            unique_pair_keys.add(row.cache_key)
            if not cached:
                unique_miss_keys.add(row.cache_key)
            materialized = replace(row, cache_hit=cached).to_dict()
            materialized["facet_id"] = candidate["facet_id"]
            materialized["manifest_order"] = candidate["manifest_order"]
            materialized["prior_bm25_rank"] = candidate["prior_bm25_rank"]
            window_rows.append(materialized)
            local_coverage.append(row.document_token_coverage_fraction)
            topic_windows[str(candidate["topic_id"])] += 1
            facet_windows[str(candidate["facet_id"])] += 1
        coverage[f"{candidate['facet_id']}:{candidate['document_id']}"] = local_coverage
    elapsed = time.perf_counter() - started
    prior_observed_pairs_per_second = prior_forward_pairs / prior_elapsed
    prior_pairs_per_second = min(
        prior_policy_rate, prior_observed_pairs_per_second
    )
    projected_inference_seconds = (
        prior_fixed_seconds + len(unique_miss_keys) / prior_pairs_per_second
    )
    enforce_preflight_ceiling(
        len(candidates), len(window_rows), projected_inference_seconds
    )
    if _cache_binding(cache) != cache_before:
        raise ValueError("score cache changed during tokenizer-only preflight")

    candidate_output = [
        {"schema_version": CANDIDATE_SCHEMA_VERSION, **row} for row in candidates
    ]
    candidate_bytes = _jsonl_bytes(candidate_output)
    window_bytes = _jsonl_bytes(window_rows)
    topic_counts = Counter(str(row["topic_id"]) for row in candidates)
    summary: dict[str, object] = {
        "query_document_pair_count": len(candidates),
        "accepted_facet_count": len(facet_counts),
        "window_count": len(window_rows),
        "cache_hit_window_count": hits,
        "cache_miss_window_count": len(window_rows) - hits,
        "unique_pair_count": len(unique_pair_keys),
        "unique_cache_miss_count": len(unique_miss_keys),
        "topic_pair_counts": dict(sorted(topic_counts.items(), key=lambda item: TOPIC_IDS.index(item[0]))),
        "facet_pair_counts": dict(sorted(facet_counts.items())),
        "topic_window_counts": dict(sorted(topic_windows.items(), key=lambda item: TOPIC_IDS.index(item[0]))),
        "facet_window_counts": dict(sorted(facet_windows.items())),
        "document_window_coverage": coverage,
    }
    runtime_evidence: dict[str, object] = {
        "policy": "prior_compatible_rocm_rate_plus_fixed_seconds_v1",
        "prior_scoring_receipt_sha256": _sha256_bytes(scoring_source),
        "prior_unique_forward_pair_count": prior_forward_pairs,
        "prior_elapsed_seconds": prior_elapsed,
        "policy_pairs_per_second": prior_policy_rate,
        "prior_pairs_per_second": prior_pairs_per_second,
        "prior_observed_pairs_per_second": prior_observed_pairs_per_second,
        "fixed_seconds": prior_fixed_seconds,
        "projected_unique_cache_miss_count": len(unique_miss_keys),
        "projected_inference_seconds": projected_inference_seconds,
        "runtime_ceiling_seconds": RUNTIME_CEILING_SECONDS,
        "prior_peak_device_memory_bytes": prior_peak_device,
        "prior_peak_host_memory_bytes": prior_peak_host,
        "tokenizer_planning_elapsed_seconds": elapsed,
    }
    source_bindings = {
        "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256_bytes(manifest_source)},
        "phase1_scoring_receipt": {"path": str(scoring_receipt_path.resolve()), "sha256": _sha256_bytes(scoring_source)},
        "phase1_preflight_receipt": {"path": str(phase1_preflight_path.resolve()), "sha256": _sha256_bytes(phase1_preflight_source)},
        "phase1_candidates": {"path": str(candidates_path.resolve()), "sha256": _sha256_bytes(phase1_source)},
        "phase1_windows": {"path": str((phase1_dir / "windows.jsonl").resolve()), "sha256": expected_windows_sha},
        "phase1_scores": {"path": str((phase1_dir / "scores.jsonl").resolve()), "sha256": expected_scores_sha},
        "gate_summary": {"path": str(gate_summary_path.resolve()), "sha256": _sha256_bytes(gate_summary_source)},
        "gates": {"path": str(gates_path.resolve()), "sha256": _sha256_bytes(gates_source)},
        "accepted_union": accepted_union_binding,
        "model_materialization_receipt": {"path": str(materialization.receipt_path), "sha256": materialization.sha256},
    }
    payload: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "tokenizer_only_preflight_complete",
        "qrels_opened": False,
        "retrieval_path_supported": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
        "model_constructed": False,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_materialization_receipt": str(materialization.receipt_path),
        "model_materialization_receipt_sha256": materialization.sha256,
        "device_probe": device_evidence,
        "runtime_evidence": runtime_evidence,
        "tokenizer": {"class": type(tokenizer).__name__, "local_files_only": True},
        "score_cache": {"context": score_cache_context().artifact_metadata, "binding": cache_before},
        "sources": source_bindings,
        "candidates_file": "candidates.jsonl",
        "candidates_sha256": _sha256_bytes(candidate_bytes),
        "windows_file": "windows.jsonl",
        "windows_sha256": _sha256_bytes(window_bytes),
        "summary": summary,
        "ceilings": {
            "exact_query_document_pair_count": PAIR_COUNT,
            "maximum_window_count": WINDOW_CEILING,
            "maximum_runtime_seconds": RUNTIME_CEILING_SECONDS,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    _exclusive_write(destination / "candidates.jsonl", candidate_bytes)
    _exclusive_write(destination / "windows.jsonl", window_bytes)
    _exclusive_write(destination / "preflight.json", _pretty_bytes(payload))
    return payload


def _float32(value: object) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("MiniLM score must be finite")
    return struct.unpack(">f", struct.pack(">f", converted))[0]


def _host_memory_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _selected_top4(
    windows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Mirror the frozen span-distinct selection used by ``aggregate_top4``."""

    ordered = sorted(
        windows,
        key=lambda row: (
            -float(row["score"]),
            int(row["document_start_token"]),
            str(row["window_id"]),
        ),
    )
    selected: list[Mapping[str, object]] = []
    covered: list[tuple[int, int]] = []
    for row in ordered:
        start = int(row["document_start_token"])
        end = int(row["document_end_token"])
        score = float(row["score"])
        if not math.isfinite(score) or start < 0 or end <= start:
            raise ValueError("window score and token span must be valid")
        overlaps = sorted(
            (max(start, left), min(end, right))
            for left, right in covered
            if min(end, right) > max(start, left)
        )
        cursor = start
        covered_count = 0
        for left, right in overlaps:
            if right <= cursor:
                continue
            covered_count += right - max(left, cursor)
            cursor = right
        if selected and end - start - covered_count < 128:
            continue
        selected.append(row)
        covered.append((start, end))
        if len(selected) == 4:
            break
    return selected


def aggregate_document_scores(
    window_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate exact query/document windows with the frozen top-four contract."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in window_rows:
        topic_id = reject_topic(row.get("topic_id"))
        facet_id = _required_text(row, "facet_id", "scored window")
        document_id = _required_text(row, "document_id", "scored window")
        grouped[(topic_id, facet_id, document_id)].append(row)

    output: list[dict[str, object]] = []
    for identity, windows in sorted(
        grouped.items(),
        key=lambda item: (
            TOPIC_IDS.index(item[0][0]),
            item[0][1],
            item[0][2],
        ),
    ):
        bindings: dict[str, object] = {}
        for field in (
            "query_sha256",
            "document_sha256",
            "model",
            "model_revision",
        ):
            values = {row.get(field) for row in windows}
            if len(values) != 1 or None in values:
                raise ValueError(f"document windows differ on required {field} binding")
            bindings[field] = next(iter(values))
        if (
            bindings["model"] != MODEL_ID
            or bindings["model_revision"] != MODEL_REVISION
        ):
            raise ValueError("document windows differ from the frozen model identity")
        selected = _selected_top4(windows)
        if not selected:
            raise ValueError("top-four aggregation requires at least one scored window")
        window_hashes = []
        for row in selected:
            window_hash = row.get("window_sha256")
            if not isinstance(window_hash, str) or len(window_hash) != 64:
                raise ValueError("selected window requires its exact text hash")
            window_hashes.append(window_hash)
        topic_id, facet_id, document_id = identity
        output.append(
            {
                "topic_id": topic_id,
                "facet_id": facet_id,
                "document_id": document_id,
                "score": aggregate_top4(windows),
                "selected_window_count": len(selected),
                "query_sha256": bindings["query_sha256"],
                "text_sha256": bindings["document_sha256"],
                "model": bindings["model"],
                "model_revision": bindings["model_revision"],
                "window_hashes": window_hashes,
            }
        )
    return output


def _validate_preflight_evidence(preflight: Mapping[str, object]) -> None:
    tokenizer = preflight.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or tokenizer.get("local_files_only") is not True
        or not isinstance(tokenizer.get("class"), str)
        or not str(tokenizer["class"]).strip()
    ):
        raise ValueError("tethered preflight tokenizer locality evidence is invalid")

    score_cache = preflight.get("score_cache")
    if not isinstance(score_cache, Mapping):
        raise ValueError("tethered preflight score-cache evidence is invalid")
    if score_cache.get("context") != score_cache_context().artifact_metadata:
        raise ValueError("tethered preflight score-cache context differs")
    cache_binding = score_cache.get("binding")
    if not isinstance(cache_binding, Mapping):
        raise ValueError("tethered preflight score-cache binding is invalid")
    state = cache_binding.get("state")
    cache_path = cache_binding.get("path")
    cache_bytes = cache_binding.get("bytes")
    cache_sha = cache_binding.get("sha256")
    if (
        state not in {"absent", "present"}
        or not isinstance(cache_path, str)
        or not Path(cache_path).is_absolute()
        or type(cache_bytes) is not int
        or cache_bytes < 0
        or (state == "absent" and (cache_bytes != 0 or cache_sha is not None))
        or (state == "present" and not isinstance(cache_sha, str))
    ):
        raise ValueError("tethered preflight score-cache binding is invalid")
    if state == "present":
        _required_sha256(cache_sha, "tethered score cache")

    sources = preflight.get("sources")
    source_names = {
        "manifest",
        "phase1_scoring_receipt",
        "phase1_preflight_receipt",
        "phase1_candidates",
        "phase1_windows",
        "phase1_scores",
        "gate_summary",
        "gates",
        "accepted_union",
        "model_materialization_receipt",
    }
    if not isinstance(sources, Mapping) or set(sources) != source_names:
        raise ValueError("tethered preflight source evidence is invalid")
    for name in source_names - {"accepted_union"}:
        binding = sources.get(name)
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("path"), str)
            or not str(binding["path"]).strip()
        ):
            raise ValueError(f"tethered preflight source {name} is invalid")
        _required_sha256(binding.get("sha256"), f"tethered source {name}")
    model_source = sources["model_materialization_receipt"]
    if (
        model_source.get("sha256")
        != preflight.get("model_materialization_receipt_sha256")
        or model_source.get("path") != preflight.get("model_materialization_receipt")
    ):
        raise ValueError("tethered preflight model receipt evidence differs")
    accepted = sources.get("accepted_union")
    if not isinstance(accepted, Mapping):
        raise ValueError("tethered preflight accepted-union source evidence is invalid")
    if (
        not isinstance(accepted.get("path"), str)
        or type(accepted.get("bytes")) is not int
        or int(accepted["bytes"]) <= 0
        or accepted.get("rows") != ACCEPTED_UNION_COUNT
        or accepted.get("topic_counts") != ACCEPTED_UNION_COUNTS
    ):
        raise ValueError("tethered preflight accepted-union source evidence is invalid")
    _required_sha256(accepted.get("sha256"), "tethered accepted union")

    device = preflight.get("device_probe")
    if not isinstance(device, Mapping):
        raise ValueError("tethered preflight ROCm device evidence is invalid")
    _validate_device_probe(device)


def _validate_runtime_evidence(
    preflight: Mapping[str, object], summary: Mapping[str, object]
) -> None:
    runtime = preflight.get("runtime_evidence")
    required = frozenset(
        {
            "policy",
            "prior_scoring_receipt_sha256",
            "prior_unique_forward_pair_count",
            "prior_elapsed_seconds",
            "policy_pairs_per_second",
            "prior_pairs_per_second",
            "prior_observed_pairs_per_second",
            "fixed_seconds",
            "projected_unique_cache_miss_count",
            "projected_inference_seconds",
            "runtime_ceiling_seconds",
            "prior_peak_device_memory_bytes",
            "prior_peak_host_memory_bytes",
            "tokenizer_planning_elapsed_seconds",
        }
    )
    if not isinstance(runtime, Mapping):
        raise ValueError("tethered preflight runtime evidence is invalid")
    _require_fields(runtime, required, "tethered preflight runtime evidence")
    if runtime.get("policy") != "prior_compatible_rocm_rate_plus_fixed_seconds_v1":
        raise ValueError("tethered preflight runtime policy differs")
    _required_sha256(
        runtime.get("prior_scoring_receipt_sha256"), "prior scoring receipt"
    )
    sources = preflight.get("sources")
    prior_source = (
        sources.get("phase1_scoring_receipt")
        if isinstance(sources, Mapping)
        else None
    )
    if (
        not isinstance(prior_source, Mapping)
        or runtime.get("prior_scoring_receipt_sha256")
        != prior_source.get("sha256")
    ):
        raise ValueError("tethered runtime evidence differs from prior scoring source")
    forward = _required_positive_int(
        runtime.get("prior_unique_forward_pair_count"), "prior forward pairs"
    )
    prior_elapsed = _required_positive_float(
        runtime.get("prior_elapsed_seconds"), "prior elapsed seconds"
    )
    policy_rate = _required_positive_float(
        runtime.get("policy_pairs_per_second"), "policy pairs per second"
    )
    observed_rate = _required_positive_float(
        runtime.get("prior_observed_pairs_per_second"),
        "prior observed pairs per second",
    )
    effective_rate = _required_positive_float(
        runtime.get("prior_pairs_per_second"), "prior pairs per second"
    )
    fixed = _required_positive_float(runtime.get("fixed_seconds"), "fixed seconds")
    projected = _required_positive_float(
        runtime.get("projected_inference_seconds"), "projected inference seconds"
    )
    planning = runtime.get("tokenizer_planning_elapsed_seconds")
    if (
        fixed != REFERENCE_FIXED_SECONDS
        or runtime.get("runtime_ceiling_seconds") != RUNTIME_CEILING_SECONDS
        or runtime.get("projected_unique_cache_miss_count")
        != summary.get("unique_cache_miss_count")
        or not math.isclose(observed_rate, forward / prior_elapsed, rel_tol=1e-12)
        or not math.isclose(effective_rate, min(policy_rate, observed_rate), rel_tol=1e-12)
        or not math.isclose(
            projected,
            fixed + int(summary["unique_cache_miss_count"]) / effective_rate,
            rel_tol=1e-12,
        )
        or projected > RUNTIME_CEILING_SECONDS
        or isinstance(planning, bool)
        or not isinstance(planning, (int, float))
        or not math.isfinite(float(planning))
        or float(planning) < 0
    ):
        raise ValueError("tethered preflight runtime evidence differs")
    _required_positive_int(
        runtime.get("prior_peak_device_memory_bytes"), "prior peak device memory"
    )
    _required_positive_int(
        runtime.get("prior_peak_host_memory_bytes"), "prior peak host memory"
    )


def _expected_topic_pairs_from_lineage(
    preflight: Mapping[str, object], facet_counts: Mapping[object, object]
) -> tuple[dict[str, int], dict[str, tuple[str, int, str, str]]]:
    """Recompute frozen topic counts from hash-bound manifest and gate sources."""

    sources = preflight.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("tethered preflight source evidence is invalid")

    loaded: dict[str, Mapping[str, object]] = {}
    for name, description in (("manifest", "deep-facet manifest"), ("gates", "gates")):
        binding = sources.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"tethered preflight source {name} is invalid")
        path = binding.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"tethered preflight source {name} is invalid")
        payload, source = _read_object(Path(path), description)
        if _sha256_bytes(source) != _required_sha256(
            binding.get("sha256"), f"tethered source {name}"
        ):
            raise ValueError(f"tethered preflight source {name} hash differs")
        loaded[name] = payload

    manifest = loaded["manifest"]
    _validate_manifest_hashes(manifest)
    if manifest.get("topic_ids") != list(TOPIC_IDS):
        raise ValueError("deep-facet manifest topic order differs")
    narratives: dict[str, str] = {}
    for row in _rows(manifest.get("topics"), "manifest topics"):
        topic_id = reject_topic(row.get("topic_id"))
        narrative = _required_text(row, "query", "manifest topic")
        if topic_id in narratives:
            raise ValueError("duplicate manifest topic")
        narratives[topic_id] = narrative
    if set(narratives) != set(TOPIC_IDS):
        raise ValueError("deep-facet manifest topics differ")

    manifest_facets: dict[str, tuple[str, int, str, str]] = {}
    for row in _rows(manifest.get("facets"), "manifest facets"):
        topic_id = reject_topic(row.get("topic_id"))
        facet_id = _required_text(row, "facet_id", "manifest facet")
        facet_query = _required_text(row, "query", "manifest facet")
        order = row.get("manifest_order")
        if type(order) is not int or order < 0:
            raise ValueError("manifest facet order must be a nonnegative integer")
        if facet_id in manifest_facets:
            raise ValueError("duplicate manifest facet")
        manifest_facets[facet_id] = (
            topic_id,
            order,
            facet_query,
            render_tethered_query(narratives[topic_id], facet_query),
        )

    gates = loaded["gates"]
    _require_fields(gates, frozenset({"schema_version", "gates"}), "gates receipt")
    if gates.get("schema_version") != "deep-facet-candidate-gate-v1":
        raise ValueError("gates receipt identity differs")
    gate_rows = _rows(gates.get("gates"), "gates")
    accepted: dict[str, tuple[str, int, str, str]] = {}
    seen: set[str] = set()
    for row in gate_rows:
        topic_id = reject_topic(row.get("topic_id"))
        facet_id = _required_text(row, "facet_id", "gate")
        status = row.get("status")
        if status not in {"accepted", "rejected"}:
            raise ValueError("gate status differs from frozen values")
        manifest_identity = manifest_facets.get(facet_id)
        if manifest_identity is None or manifest_identity[:2] != (
            topic_id,
            row.get("manifest_order"),
        ):
            raise ValueError("gate facet lineage differs from manifest")
        if facet_id in seen:
            raise ValueError("duplicate gate facet")
        seen.add(facet_id)
        if status == "accepted":
            accepted[facet_id] = manifest_identity
    if (
        len(gate_rows) != 25
        or seen != set(manifest_facets)
        or len(accepted) != ACCEPTED_FACET_COUNT
        or len(gate_rows) - len(accepted) != 1
    ):
        raise ValueError("gate required exact facet counts differ")
    if set(map(str, facet_counts)) != set(accepted):
        raise ValueError("preflight facet counts differ from accepted gate lineage")

    expected: Counter[str] = Counter()
    for facet_id, (topic_id, _order, _facet_query, _query) in accepted.items():
        pair_count = facet_counts.get(facet_id)
        if type(pair_count) is not int or pair_count != 200:
            raise ValueError("accepted facets require exactly 200 candidate pairs")
        expected[topic_id] += pair_count
    return (
        {topic_id: expected[topic_id] for topic_id in TOPIC_IDS},
        accepted,
    )


def _validate_generated_lineage(
    output: Path,
    preflight: Mapping[str, object],
    summary: Mapping[str, object],
    accepted: Mapping[str, tuple[str, int, str, str]],
    expected_topic_counts: Mapping[str, int],
) -> None:
    candidate_path = output / str(preflight["candidates_file"])
    window_path = output / str(preflight["windows_file"])
    candidate_source = candidate_path.read_bytes()
    window_source = window_path.read_bytes()
    if _sha256_bytes(candidate_source) != preflight.get("candidates_sha256"):
        raise ValueError("generated candidate ledger hash differs")
    if _sha256_bytes(window_source) != preflight.get("windows_sha256"):
        raise ValueError("planned window ledger hash differs")

    candidates = _read_jsonl_source(candidate_source, "tethered MiniLM candidates")
    windows = _read_jsonl_source(window_source, "tethered MiniLM windows")
    if len(candidates) != PAIR_COUNT or len(windows) != summary.get("window_count"):
        raise ValueError("generated candidate or window ledger count differs")

    candidate_by_identity: dict[
        tuple[str, str, str], Mapping[str, object]
    ] = {}
    candidate_topics: Counter[str] = Counter()
    candidate_facets: Counter[str] = Counter()
    for candidate in candidates:
        topic_id = reject_topic(candidate.get("topic_id"))
        facet_id = _required_text(candidate, "facet_id", "tethered candidate")
        document_id = _required_text(
            candidate, "document_id", "tethered candidate"
        )
        facet = accepted.get(facet_id)
        query = _required_text(candidate, "query", "tethered candidate")
        facet_query = _required_text(
            candidate, "facet_query", "tethered candidate"
        )
        text = _required_text(candidate, "text", "tethered candidate")
        manifest_order = candidate.get("manifest_order")
        if (
            candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION
            or facet is None
            or type(manifest_order) is not int
            or (topic_id, manifest_order, facet_query, query) != facet
            or candidate.get("facet_query_sha256") != _sha256_text(facet_query)
            or candidate.get("query_sha256") != _sha256_text(query)
            or candidate.get("text_sha256") != _sha256_text(text)
        ):
            raise ValueError("tethered candidate lineage differs from accepted facets")
        identity = (topic_id, facet_id, document_id)
        if identity in candidate_by_identity:
            raise ValueError("duplicate tethered candidate lineage identity")
        candidate_by_identity[identity] = candidate
        candidate_topics[topic_id] += 1
        candidate_facets[facet_id] += 1
    if (
        dict(candidate_topics) != dict(expected_topic_counts)
        or dict(candidate_topics) != summary.get("topic_pair_counts")
        or dict(candidate_facets) != summary.get("facet_pair_counts")
    ):
        raise ValueError("tethered candidate lineage population differs")

    window_identities: set[tuple[str, str, str]] = set()
    window_topics: Counter[str] = Counter()
    window_facets: Counter[str] = Counter()
    for window in windows:
        topic_id = reject_topic(window.get("topic_id"))
        facet_id = _required_text(window, "facet_id", "planned window")
        document_id = _required_text(window, "document_id", "planned window")
        identity = (topic_id, facet_id, document_id)
        candidate = candidate_by_identity.get(identity)
        if (
            candidate is None
            or window.get("query") != candidate.get("query")
            or window.get("query_sha256") != candidate.get("query_sha256")
            or window.get("document_sha256") != candidate.get("text_sha256")
        ):
            raise ValueError("tethered candidate and window identities differ")
        window_identities.add(identity)
        window_topics[topic_id] += 1
        window_facets[facet_id] += 1
    if (
        window_identities != set(candidate_by_identity)
        or dict(window_topics) != summary.get("topic_window_counts")
        or dict(window_facets) != summary.get("facet_window_counts")
    ):
        raise ValueError("tethered candidate and window identities differ")


def verify_preflight(
    preflight_source: Path | Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on the Task 1 receipt before cache or model construction."""

    if isinstance(preflight_source, Mapping):
        preflight = dict(preflight_source)
        output = None
    else:
        preflight_path = Path(preflight_source)
        preflight, _ = _read_object(preflight_path, "tethered MiniLM preflight")
        output = preflight_path.parent

    # The topic firewall deliberately precedes all other dependency access.
    topic_values: list[object] = []
    summary = preflight.get("summary")
    if isinstance(summary, Mapping):
        counts = summary.get("topic_pair_counts")
        if isinstance(counts, Mapping):
            topic_values.extend(counts)
    raw_topics = preflight.get("topic_ids")
    if isinstance(raw_topics, Sequence) and not isinstance(raw_topics, (str, bytes)):
        topic_values.extend(raw_topics)
    for topic_id in topic_values:
        reject_topic(topic_id)

    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or preflight.get("status") != "tokenizer_only_preflight_complete"
        or preflight.get("qrels_opened") is not False
        or preflight.get("retrieval_path_supported") is not False
        or preflight.get("network_access_supported") is not False
        or preflight.get("hosted_inference_supported") is not False
        or preflight.get("model_constructed") is not False
        or preflight.get("model") != MODEL_ID
        or preflight.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("tethered MiniLM preflight differs from the frozen contract")
    if (
        preflight.get("windows_file") != "windows.jsonl"
        or preflight.get("candidates_file") != "candidates.jsonl"
    ):
        raise ValueError("tethered MiniLM artifact filenames must be fixed local basenames")
    _required_sha256(preflight.get("windows_sha256"), "tethered windows")
    _required_sha256(preflight.get("candidates_sha256"), "tethered candidates")
    _required_sha256(
        preflight.get("model_materialization_receipt_sha256"),
        "model materialization receipt",
    )
    if not isinstance(preflight.get("model_materialization_receipt"), str):
        raise ValueError("model materialization receipt path is required")
    if not isinstance(summary, Mapping):
        raise ValueError("tethered MiniLM preflight summary is required")
    _require_fields(
        summary,
        frozenset(
            {
                "query_document_pair_count",
                "accepted_facet_count",
                "window_count",
                "cache_hit_window_count",
                "cache_miss_window_count",
                "unique_pair_count",
                "unique_cache_miss_count",
                "topic_pair_counts",
                "facet_pair_counts",
                "topic_window_counts",
                "facet_window_counts",
                "document_window_coverage",
            }
        ),
        "tethered preflight summary",
    )
    if (
        summary.get("query_document_pair_count") != PAIR_COUNT
        or type(summary.get("window_count")) is not int
        or int(summary["window_count"]) > WINDOW_CEILING
        or type(summary.get("unique_pair_count")) is not int
        or type(summary.get("unique_cache_miss_count")) is not int
        or summary.get("accepted_facet_count") != ACCEPTED_FACET_COUNT
    ):
        raise ValueError("tethered MiniLM preflight counts differ from the frozen contract")
    topic_counts = summary.get("topic_pair_counts")
    facet_counts = summary.get("facet_pair_counts")
    topic_windows = summary.get("topic_window_counts")
    facet_windows = summary.get("facet_window_counts")
    coverage = summary.get("document_window_coverage")
    window_count = int(summary["window_count"])
    hit_windows = summary.get("cache_hit_window_count")
    miss_windows = summary.get("cache_miss_window_count")
    lineage = (
        _expected_topic_pairs_from_lineage(preflight, facet_counts)
        if isinstance(facet_counts, Mapping)
        else None
    )
    expected_topic_counts = lineage[0] if lineage is not None else None
    if (
        not isinstance(topic_counts, Mapping)
        or set(topic_counts) != set(TOPIC_IDS)
        or dict(topic_counts) != expected_topic_counts
        or not isinstance(facet_counts, Mapping)
        or len(facet_counts) != ACCEPTED_FACET_COUNT
        or set(facet_counts.values()) != {200}
        or not isinstance(topic_windows, Mapping)
        or set(topic_windows) != set(TOPIC_IDS)
        or any(type(value) is not int or value < 0 for value in topic_windows.values())
        or sum(topic_windows.values()) != window_count
        or not isinstance(facet_windows, Mapping)
        or set(facet_windows) != set(facet_counts)
        or any(type(value) is not int or value < 1 for value in facet_windows.values())
        or sum(facet_windows.values()) != window_count
        or type(hit_windows) is not int
        or type(miss_windows) is not int
        or hit_windows < 0
        or miss_windows < 0
        or hit_windows + miss_windows != window_count
        or window_count < PAIR_COUNT
        or not 0 <= int(summary["unique_cache_miss_count"]) <= int(summary["unique_pair_count"])
        or int(summary["unique_pair_count"]) > window_count
        or not isinstance(coverage, Mapping)
        or len(coverage) != PAIR_COUNT
    ):
        raise ValueError("tethered MiniLM preflight population differs from the frozen contract")
    for values in coverage.values():
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in values
            )
        ):
            raise ValueError("tethered preflight document coverage is invalid")
    coverage_documents: Counter[str] = Counter()
    coverage_windows: Counter[str] = Counter()
    for identity, values in coverage.items():
        matches = [
            str(facet_id)
            for facet_id in facet_counts
            if isinstance(identity, str)
            and identity.startswith(f"{facet_id}:")
            and len(identity) > len(str(facet_id)) + 1
        ]
        if len(matches) != 1:
            raise ValueError("tethered preflight document coverage identity is invalid")
        coverage_documents[matches[0]] += 1
        coverage_windows[matches[0]] += len(values)
    if (
        dict(coverage_documents) != dict(facet_counts)
        or dict(coverage_windows) != dict(facet_windows)
        or sum(coverage_windows.values()) != window_count
    ):
        raise ValueError("tethered preflight coverage population differs")
    ceilings = preflight.get("ceilings")
    if (
        not isinstance(ceilings, Mapping)
        or ceilings.get("exact_query_document_pair_count") != PAIR_COUNT
        or ceilings.get("maximum_window_count") != WINDOW_CEILING
        or ceilings.get("maximum_runtime_seconds") != RUNTIME_CEILING_SECONDS
    ):
        raise ValueError("tethered MiniLM preflight ceilings differ from the frozen contract")
    _validate_preflight_evidence(preflight)
    _validate_runtime_evidence(preflight, summary)
    if output is not None and lineage is not None:
        _validate_generated_lineage(
            output, preflight, summary, lineage[1], lineage[0]
        )
    return preflight


class _LocalMiniLMRunner:
    def __init__(self, model_receipt: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        verified = load_verified_materialization(model_receipt)
        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError("ROCm MiniLM scoring requires an available torch cuda device")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            verified.snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            verified.snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float32,
        ).float().eval().to("cuda")
        torch.cuda.reset_peak_memory_stats()

    def score(self, rows: Sequence[Mapping[str, object]]) -> list[float]:
        encoded = self.tokenizer(
            [str(row["query"]) for row in rows],
            [str(row["window_text"]) for row in rows],
            padding=True,
            truncation=False,
            max_length=PAIR_MAX_TOKENS,
            return_tensors="pt",
        )
        inputs = {name: tensor.to("cuda") for name, tensor in encoded.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits.detach().float().cpu().tolist()
        if len(logits) != len(rows) or any(
            not isinstance(value, list) or len(value) != 1 for value in logits
        ):
            raise ValueError("MiniLM logits must have shape (batch, 1)")
        return [_float32(value[0]) for value in logits]

    @property
    def peak_device_memory_bytes(self) -> int:
        return int(self.torch.cuda.max_memory_allocated())


def run_local_scoring(
    preflight_path: Path,
    cache_root: Path = SCORE_CACHE_ROOT,
    *,
    runner: object | None = None,
) -> dict[str, object]:
    """Score only frozen cache misses locally and create bound score artifacts."""

    path = Path(preflight_path)
    output = path.parent
    for filename in (
        "scoring_reservation.json",
        "scores.jsonl",
        "document_scores.jsonl",
        "scoring_receipt.json",
    ):
        if (output / filename).exists():
            raise FileExistsError(f"create-only scoring output exists: {filename}")
    preflight = verify_preflight(path)
    preflight_bytes = path.read_bytes()
    windows_path = output / "windows.jsonl"
    candidates_path = output / "candidates.jsonl"
    window_bytes = windows_path.read_bytes()
    candidate_bytes = candidates_path.read_bytes()
    if (
        _sha256_bytes(window_bytes) != preflight.get("windows_sha256")
        or _sha256_bytes(candidate_bytes) != preflight.get("candidates_sha256")
    ):
        raise ValueError("window or candidate bytes differ from tethered preflight")
    windows = _read_jsonl_source(window_bytes, "tethered MiniLM windows")
    if len(windows) != preflight["summary"]["window_count"]:  # type: ignore[index]
        raise ValueError("window rows differ from tethered preflight count")
    candidates = _read_jsonl_source(candidate_bytes, "tethered MiniLM candidates")
    if len(candidates) != PAIR_COUNT:
        raise ValueError("candidate rows differ from the exact 4,800-pair contract")
    for row in windows:
        reject_topic(row.get("topic_id"))
    window_identities = {
        (
            str(row.get("topic_id")),
            str(row.get("facet_id")),
            str(row.get("document_id")),
        )
        for row in windows
    }
    candidate_identities: set[tuple[str, str, str]] = set()
    for row in candidates:
        topic_id = reject_topic(row.get("topic_id"))
        facet_id = _required_text(row, "facet_id", "tethered candidate")
        document_id = _required_text(row, "document_id", "tethered candidate")
        if row.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
            raise ValueError("tethered candidate schema differs")
        candidate_identities.add((topic_id, facet_id, document_id))
    if len(candidate_identities) != PAIR_COUNT:
        raise ValueError("tethered candidates require 4,800 unique identities")
    if candidate_identities != window_identities:
        raise ValueError("tethered candidate and window identities differ")

    cache = GlobalScoreCache(Path(cache_root), score_cache_context())
    if _cache_binding(cache) != preflight["score_cache"]["binding"]:  # type: ignore[index]
        raise ValueError("score-cache binding changed after tethered MiniLM preflight")
    model_receipt = Path(str(preflight.get("model_materialization_receipt")))
    verified = load_verified_materialization(model_receipt)
    if verified.sha256 != preflight.get("model_materialization_receipt_sha256"):
        raise ValueError("model receipt differs from tethered MiniLM preflight")
    by_key: dict[str, Mapping[str, object]] = {}
    planned_miss_keys: set[str] = set()
    for row in windows:
        query = _required_text(row, "query", "planned window")
        text = _required_text(row, "window_text", "planned window")
        key = _required_text(row, "cache_key", "planned window")
        if cache.cache_key(query_text=query, text=text) != key:  # type: ignore[attr-defined]
            raise ValueError("planned window cache key differs from exact content")
        by_key.setdefault(key, row)
        if row.get("cache_hit") is False:
            planned_miss_keys.add(key)
    actual_misses = [
        row
        for key, row in sorted(by_key.items())
        if cache.get(query_text=str(row["query"]), text=str(row["window_text"])) is None  # type: ignore[attr-defined]
    ]
    actual_miss_keys = {str(row["cache_key"]) for row in actual_misses}
    if (
        actual_miss_keys != planned_miss_keys
        or len(actual_miss_keys) != preflight["summary"]["unique_cache_miss_count"]  # type: ignore[index]
    ):
        raise ValueError("score-cache misses changed after tethered MiniLM preflight")

    reservation_bytes = _pretty_bytes(
        {
            "schema_version": SCORING_RECEIPT_SCHEMA_VERSION,
            "status": "reserved",
            "preflight_sha256": _sha256_bytes(preflight_bytes),
            "planned_unique_forward_pair_count": len(actual_misses),
            "hard_ceiling_seconds": RUNTIME_CEILING_SECONDS,
            "qrels_opened": False,
            "automatic_retry": False,
        }
    )
    _exclusive_write(output / "scoring_reservation.json", reservation_bytes)
    started = time.perf_counter()
    scorer = runner
    if actual_misses and scorer is None:
        scorer = _LocalMiniLMRunner(model_receipt)
    for start in range(0, len(actual_misses), BATCH_SIZE):
        if time.perf_counter() - started >= RUNTIME_CEILING_SECONDS:
            raise RuntimeError("tethered MiniLM scoring exceeded the 600-second ceiling; no retry")
        batch = actual_misses[start : start + BATCH_SIZE]
        values = scorer.score(batch)  # type: ignore[union-attr]
        if len(values) != len(batch):
            raise ValueError("MiniLM runner returned an unexpected score count")
        cache.add_many(  # type: ignore[attr-defined]
            (str(row["query"]), str(row["window_text"]), _float32(score))
            for row, score in zip(batch, values, strict=True)
        )
        if time.perf_counter() - started >= RUNTIME_CEILING_SECONDS:
            raise RuntimeError("tethered MiniLM scoring exceeded the 600-second ceiling; no retry")

    score_rows: list[dict[str, object]] = []
    for row in windows:
        score = cache.get(query_text=str(row["query"]), text=str(row["window_text"]))  # type: ignore[attr-defined]
        if score is None:
            raise ValueError("score cache does not cover a frozen tethered window")
        score_rows.append(
            {
                **row,
                "schema_version": SCORE_SCHEMA_VERSION,
                "score": _float32(score),
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "score_representation": "raw_logits",
                "inference_dtype": "float32",
            }
        )
    document_rows = aggregate_document_scores(score_rows)
    if len(document_rows) != PAIR_COUNT:
        raise ValueError("document aggregation requires exactly 4,800 rows")
    bound_document_rows = [
        {"schema_version": DOCUMENT_SCORE_SCHEMA_VERSION, **row}
        for row in document_rows
    ]
    score_bytes = _jsonl_bytes(score_rows)
    document_bytes = _jsonl_bytes(bound_document_rows)
    if time.perf_counter() - started >= RUNTIME_CEILING_SECONDS:
        raise RuntimeError("tethered MiniLM scoring exceeded the 600-second ceiling; no retry")
    _exclusive_write(output / "scores.jsonl", score_bytes)
    _exclusive_write(output / "document_scores.jsonl", document_bytes)
    elapsed = time.perf_counter() - started
    if elapsed >= RUNTIME_CEILING_SECONDS:
        raise RuntimeError("tethered MiniLM scoring exceeded the 600-second ceiling; no retry")
    receipt: dict[str, object] = {
        "schema_version": SCORING_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "qrels_opened": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preflight_sha256": _sha256_bytes(preflight_bytes),
        "scoring_reservation_sha256": _sha256_bytes(reservation_bytes),
        "windows_sha256": _sha256_bytes(window_bytes),
        "scores_sha256": _sha256_bytes(score_bytes),
        "document_scores_sha256": _sha256_bytes(document_bytes),
        "planned_window_count": len(windows),
        "completed_window_count": len(score_rows),
        "document_score_count": len(document_rows),
        "unique_forward_pair_count": len(actual_misses),
        "cache_hit_count": len(by_key) - len(actual_misses),
        "cache_reuse_pair_count": len(by_key) - len(actual_misses),
        "elapsed_seconds": elapsed,
        "hard_ceiling_seconds": RUNTIME_CEILING_SECONDS,
        "peak_device_memory_bytes": int(
            getattr(scorer, "peak_device_memory_bytes", 0)
        ),
        "peak_host_memory_bytes": _host_memory_bytes(),
        "device": "cuda",
        "execution_backend": "rocm",
        "score_cache_path": str(cache.path),  # type: ignore[attr-defined]
    }
    _exclusive_write(output / "scoring_receipt.json", _pretty_bytes(receipt))
    return receipt


def verify_scoring(preflight_path: Path) -> dict[str, object]:
    """Verify create-only scoring artifacts remain bound to the Task 1 plan."""

    path = Path(preflight_path)
    preflight = verify_preflight(path)
    output = path.parent
    receipt, _ = _read_object(output / "scoring_receipt.json", "scoring receipt")
    reservation, reservation_bytes = _read_object(
        output / "scoring_reservation.json", "scoring reservation"
    )
    candidate_bytes = (output / "candidates.jsonl").read_bytes()
    window_bytes = (output / "windows.jsonl").read_bytes()
    score_bytes = (output / "scores.jsonl").read_bytes()
    document_bytes = (output / "document_scores.jsonl").read_bytes()
    candidates = _read_jsonl_source(candidate_bytes, "tethered MiniLM candidates")
    windows = _read_jsonl_source(window_bytes, "tethered MiniLM windows")
    score_rows = _read_jsonl_source(score_bytes, "tethered MiniLM scores")
    document_rows = _read_jsonl_source(
        document_bytes, "tethered MiniLM document scores"
    )
    if _sha256_bytes(candidate_bytes) != preflight.get("candidates_sha256"):
        raise ValueError("current candidates differ from the tethered preflight")
    if _sha256_bytes(window_bytes) != preflight.get("windows_sha256"):
        raise ValueError("current planned windows differ from the tethered preflight")
    candidate_identities: set[tuple[str, str, str]] = set()
    candidate_topics: Counter[str] = Counter()
    candidate_facets: Counter[str] = Counter()
    windows_by_identity: dict[
        tuple[str, str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    for window in windows:
        windows_by_identity[
            (
                str(window.get("topic_id")),
                str(window.get("facet_id")),
                str(window.get("document_id")),
            )
        ].append(window)
    for candidate in candidates:
        topic_id = reject_topic(candidate.get("topic_id"))
        facet_id = _required_text(candidate, "facet_id", "tethered candidate")
        document_id = _required_text(
            candidate, "document_id", "tethered candidate"
        )
        if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
            raise ValueError("tethered candidate schema differs")
        identity = (topic_id, facet_id, document_id)
        query = _required_text(candidate, "query", "tethered candidate")
        text = _required_text(candidate, "text", "tethered candidate")
        rank = candidate.get("rank")
        candidate_windows = windows_by_identity.get(identity, [])
        if (
            candidate.get("query_sha256") != _sha256_text(query)
            or candidate.get("text_sha256") != _sha256_text(text)
            or type(rank) is not int
            or rank < 1
            or not candidate_windows
            or any(
                window.get("query") != query
                or window.get("query_sha256") != candidate.get("query_sha256")
                or window.get("document_sha256") != candidate.get("text_sha256")
                or window.get("rank") != rank
                for window in candidate_windows
            )
        ):
            raise ValueError("tethered candidate query/text lineage differs from windows")
        candidate_identities.add(identity)
        candidate_topics[topic_id] += 1
        candidate_facets[facet_id] += 1
    window_identities = {
        (
            str(window.get("topic_id")),
            str(window.get("facet_id")),
            str(window.get("document_id")),
        )
        for window in windows
    }
    document_identities = {
        (
            str(document.get("topic_id")),
            str(document.get("facet_id")),
            str(document.get("document_id")),
        )
        for document in document_rows
    }
    summary = preflight["summary"]
    if (
        len(candidates) != PAIR_COUNT
        or len(candidate_identities) != PAIR_COUNT
        or candidate_identities != window_identities
        or candidate_identities != document_identities
        or dict(candidate_topics) != summary["topic_pair_counts"]  # type: ignore[index]
        or dict(candidate_facets) != summary["facet_pair_counts"]  # type: ignore[index]
    ):
        raise ValueError("tethered candidate and window identities differ")
    if len(windows) != len(score_rows):
        raise ValueError("scoring rows do not exactly cover planned windows")
    lineage_fields = (
        "topic_id",
        "facet_id",
        "document_id",
        "window_id",
        "query_sha256",
        "document_sha256",
        "window_sha256",
        "cache_key",
    )
    for window, score_row in zip(windows, score_rows, strict=True):
        if any(window.get(field) != score_row.get(field) for field in lineage_fields):
            raise ValueError("score row lineage differs from its planned window")
        score = score_row.get("score")
        if (
            score_row.get("schema_version") != SCORE_SCHEMA_VERSION
            or score_row.get("model") != MODEL_ID
            or score_row.get("model_revision") != MODEL_REVISION
            or score_row.get("score_representation") != "raw_logits"
            or score_row.get("inference_dtype") != "float32"
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or _float32(score) != float(score)
        ):
            raise ValueError("score row differs from the frozen float32 contract")
        planned_payload = {
            key: value for key, value in window.items() if key != "schema_version"
        }
        scored_payload = {
            key: value
            for key, value in score_row.items()
            if key
            not in {
                "schema_version",
                "score",
                "model",
                "model_revision",
                "score_representation",
                "inference_dtype",
            }
        }
        if scored_payload != planned_payload:
            raise ValueError("score row payload differs from its exact planned window")
    recomputed = [
        {"schema_version": DOCUMENT_SCORE_SCHEMA_VERSION, **row}
        for row in aggregate_document_scores(score_rows)
    ]
    if _jsonl_bytes(recomputed) != document_bytes or len(document_rows) != PAIR_COUNT:
        raise ValueError("document aggregation differs from scored windows")
    preflight_sha = _sha256_bytes(path.read_bytes())
    planned_misses = summary["unique_cache_miss_count"]  # type: ignore[index]
    unique_pairs = summary["unique_pair_count"]  # type: ignore[index]
    elapsed = receipt.get("elapsed_seconds")
    if (
        reservation.get("schema_version") != SCORING_RECEIPT_SCHEMA_VERSION
        or reservation.get("status") != "reserved"
        or reservation.get("qrels_opened") is not False
        or reservation.get("automatic_retry") is not False
        or reservation.get("preflight_sha256") != preflight_sha
        or reservation.get("hard_ceiling_seconds") != RUNTIME_CEILING_SECONDS
        or reservation.get("planned_unique_forward_pair_count") != planned_misses
        or receipt.get("schema_version") != SCORING_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("qrels_opened") is not False
        or receipt.get("network_access_supported") is not False
        or receipt.get("hosted_inference_supported") is not False
        or receipt.get("model") != MODEL_ID
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("device") != "cuda"
        or receipt.get("execution_backend") != "rocm"
        or receipt.get("hard_ceiling_seconds") != RUNTIME_CEILING_SECONDS
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or not 0 <= float(elapsed) < RUNTIME_CEILING_SECONDS
        or receipt.get("preflight_sha256") != preflight_sha
        or receipt.get("scoring_reservation_sha256") != _sha256_bytes(reservation_bytes)
        or receipt.get("windows_sha256") != preflight.get("windows_sha256")
        or receipt.get("scores_sha256") != _sha256_bytes(score_bytes)
        or receipt.get("document_scores_sha256") != _sha256_bytes(document_bytes)
        or len(score_bytes.splitlines()) != preflight["summary"]["window_count"]  # type: ignore[index]
        or len(document_bytes.splitlines()) != PAIR_COUNT
        or receipt.get("planned_window_count") != len(windows)
        or receipt.get("completed_window_count") != len(score_rows)
        or receipt.get("document_score_count") != len(document_rows)
        or receipt.get("unique_forward_pair_count")
        != reservation.get("planned_unique_forward_pair_count")
        or receipt.get("cache_hit_count") != unique_pairs - planned_misses
        or receipt.get("cache_reuse_pair_count") != unique_pairs - planned_misses
    ):
        raise ValueError("scoring artifacts differ from the frozen receipt")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--manifest", required=True, type=Path)
    preflight.add_argument("--phase1", required=True, type=Path)
    preflight.add_argument("--gate", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    preflight.add_argument("--cache-root", type=Path, default=SCORE_CACHE_ROOT)
    score = subparsers.add_parser("score")
    score.add_argument("--preflight", required=True, type=Path)
    score.add_argument("--cache-root", type=Path, default=SCORE_CACHE_ROOT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--preflight", required=True, type=Path)
    return parser


def build_argument_parser() -> argparse.ArgumentParser:
    """Backward-compatible parser name retained for the Task 1 CLI."""

    return build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = create_preflight(
            {
                "manifest": args.manifest,
                "phase1": args.phase1,
                "gate": args.gate,
                "cache_root": args.cache_root,
            },
            args.output,
        )
    elif args.command == "score":
        result = run_local_scoring(args.preflight, cache_root=args.cache_root)
    else:
        result = verify_scoring(args.preflight)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
