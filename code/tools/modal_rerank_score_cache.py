"""Run the schema-v2 reranker score-cache builder on a persistent Modal GPU.

This is a thin orchestration layer around ``trec_rag.rerank_score_cache``.  It
does not implement scoring, and the document/window JSONL files it produces
have exactly the same schema as a local invocation of that module.

The input is a zstd-compressed tar archive uploaded to the Modal Volume.  The
archive must have one top-level ``workspace/`` directory containing the repo,
the 22-topic retrieval cache, config, and data submodules.  Do not put local
reranker score-cache files in the archive; this runner also removes that path
from a freshly extracted workspace before linking it to an empty, run-scoped
Modal cache.  That makes a fresh run fully CUDA-produced while still allowing
an interrupted run to resume from Modal's persistent scores.

Typical invocation (shown for documentation; importing this file launches
nothing)::

    uvx modal volume create trec-rag-rerank-score-cache-v2
    uvx modal volume put trec-rag-rerank-score-cache-v2 \
      workspace.tar.zst inputs/workspace.tar.zst
    uvx modal run --detach code/tools/modal_rerank_score_cache.py::main \
      --run-id rag25-hits1000-20260710 \
      --input-sha256 <sha256-of-workspace.tar.zst>

Set ``TREC_RAG_MODAL_APP_NAME`` or ``TREC_RAG_MODAL_VOLUME_NAME`` before the
``modal run`` command to override the stable defaults.  Use ``--resume`` only
for the same run id and input digest.  Results, the global schema-v2 cache, and
``runtime_status.json`` remain under ``runs/<run-id>/`` in the Volume.

After scoring completes, verify that the persistent global cache alone can
recreate semantically identical artifacts without loading or invoking a model::

    uvx modal run --detach code/tools/modal_rerank_score_cache.py::verify \
      --layout run-scoped \
      --run-id rag25-hits1000-20260710 \
      --verification-id warm-cache-v1

For the original one-off Volume layout, use ``--layout legacy`` and omit
``--run-id``.  Select that named Volume with
``TREC_RAG_MODAL_VOLUME_NAME=trec-rag-raw-logit-hits1000-20260710``.

The score function is capped at one container and also writes a Volume-local
exclusive marker.  The marker is defense in depth, not a distributed lock
across independent Modal app deployments or Volume snapshots.  Resume rejects
an active status.  After separately auditing and changing a crashed run to a
non-active status, ``--resume --recover-stale-lock`` archives and replaces a
same-run, same-input stale marker.  Resume validates all complete JSONL rows;
it archives and truncates only a non-newline final fragment before continuing.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import modal


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from trec_rag.artifact_semantics import compare_artifact_rows, load_jsonl_rows  # noqa: E402
from trec_rag.run_integrity import (  # noqa: E402
    ExclusiveFileLock,
    ThreadSafeJsonStatusWriter,
    validate_or_repair_final_jsonl,
)


APP_NAME = os.environ.get(
    "TREC_RAG_MODAL_APP_NAME",
    "trec-rag-rerank-score-cache-v2",
)
VOLUME_NAME = os.environ.get(
    "TREC_RAG_MODAL_VOLUME_NAME",
    "trec-rag-rerank-score-cache-v2",
)
VOLUME_MOUNT = "/volume"

TOPICS = (
    "14",
    "31",
    "37",
    "58",
    "72",
    "84",
    "144",
    "161",
    "200",
    "213",
    "219",
    "224",
    "225",
    "233",
    "273",
    "300",
    "407",
    "477",
    "499",
    "515",
    "707",
    "897",
)

EXPECTED_DOCUMENT_ROWS_BY_TOPIC = {topic: 1000 for topic in TOPICS}
EXPECTED_WINDOW_ROWS_BY_TOPIC = {
    "14": 6517,
    "31": 6071,
    "37": 10427,
    "58": 9538,
    "72": 7030,
    "84": 9585,
    "144": 12195,
    "161": 14425,
    "200": 6925,
    "213": 12664,
    "219": 12797,
    "224": 8758,
    "225": 5086,
    "233": 8103,
    "273": 15635,
    "300": 21688,
    "407": 8276,
    "477": 15736,
    "499": 5645,
    "515": 11533,
    "707": 9133,
    "897": 9389,
}

EXPECTED_DOCUMENT_ROWS = sum(EXPECTED_DOCUMENT_ROWS_BY_TOPIC.values())
EXPECTED_WINDOW_ROWS = sum(EXPECTED_WINDOW_ROWS_BY_TOPIC.values())

DOCUMENT_ARTIFACT_NAME = (
    "st_crossencoder_longctx_hits1000_"
    "mixedbread_ai__mxbai_rerank_base_v2_ctx32768_buf512_"
    "raw_logits_artifact_v2_scores.jsonl"
)
WINDOW_ARTIFACT_NAME = (
    "st_chunk_eval_hits1000_"
    "mixedbread_ai__mxbai_rerank_base_v2_ml1024_cm3500_ov350_"
    "raw_logits_artifact_v2_scores.jsonl"
)
LEGACY_DOCUMENT_ARTIFACT_NAME = (
    "st_crossencoder_longctx_hits1000_"
    "mixedbread_ai__mxbai_rerank_base_v2_ctx32768_buf512_"
    "raw_logits_modal_a100_scores.jsonl"
)
LEGACY_WINDOW_ARTIFACT_NAME = (
    "st_chunk_eval_hits1000_"
    "mixedbread_ai__mxbai_rerank_base_v2_ml1024_cm3500_ov350_"
    "raw_logits_modal_a100_scores.jsonl"
)

PINNED_PACKAGES = (
    "huggingface-hub==1.22.0",
    "numpy==2.5.1",
    "PyYAML==6.0.3",
    "safetensors==0.8.0",
    "semantic-text-splitter==0.32.0",
    "sentence-transformers==5.6.0",
    "tokenizers==0.22.2",
    "torch==2.9.1",
    "transformers==5.13.0",
    "zstandard==0.25.0",
)
VERSIONED_DISTRIBUTIONS = tuple(
    requirement.split("==", 1)[0] for requirement in PINNED_PACKAGES
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*PINNED_PACKAGES)
    .add_local_python_source(
        "trec_rag.artifact_semantics",
        "trec_rag.run_integrity",
        copy=True,
    )
    .env(
        {
            # Model downloads and score artifacts both survive container exit.
            "HF_HOME": f"{VOLUME_MOUNT}/model-cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=32768,
    timeout=10800,
    max_containers=1,
    volumes={VOLUME_MOUNT: volume},
)
def score_all_candidates(
    run_id: str,
    input_sha256: str,
    selected_app_name: str,
    selected_volume_name: str,
    archive_relative_path: str = "inputs/workspace.tar.zst",
    resume: bool = False,
    recover_stale_lock: bool = False,
) -> dict[str, object]:
    """Score all 1,000 BM25 candidates for the fixed 22-topic dev set."""

    import hashlib
    import importlib.metadata
    import json
    import math
    from pathlib import Path, PurePosixPath
    import re
    import shutil
    import subprocess
    import sys
    import tarfile
    import threading
    import time
    import traceback

    import torch
    import zstandard

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, dot, underscore, or hyphen (maximum 128 characters)"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise ValueError("input_sha256 must be a lowercase SHA-256 hex digest")
    if recover_stale_lock and not resume:
        raise ValueError("recover_stale_lock is only valid with resume")
    if not selected_app_name.strip() or not selected_volume_name.strip():
        raise ValueError("selected Modal app and Volume names must not be empty")

    volume_root = Path(VOLUME_MOUNT)

    def safe_volume_path(relative_path: str) -> Path:
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or not pure_path.parts
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise ValueError(f"unsafe Volume-relative path: {relative_path!r}")
        candidate = volume_root.joinpath(*pure_path.parts)
        if not candidate.resolve().is_relative_to(volume_root.resolve()):
            raise ValueError(
                f"Volume-relative path escapes its mount: {relative_path!r}"
            )
        return candidate

    archive = safe_volume_path(archive_relative_path)
    if not archive_relative_path.startswith("inputs/"):
        raise ValueError("archive_relative_path must be under inputs/")
    if not archive.is_file():
        raise FileNotFoundError(archive)

    run_root = safe_volume_path(f"runs/{run_id}")
    workspace = run_root / "workspace"
    cache_root = run_root / "cache" / "reranker" / "score_cache"
    results = run_root / "results"
    status_path = run_root / "runtime_status.json"
    marker_path = workspace / ".input_sha256"
    document_path = results / DOCUMENT_ARTIFACT_NAME
    window_path = results / WINDOW_ARTIFACT_NAME

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def line_count(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("rb") as source:
            return sum(1 for _ in source)

    def cache_inventory() -> dict[str, object]:
        files = sorted(cache_root.rglob("*.jsonl")) if cache_root.exists() else []
        return {
            "rows": sum(line_count(path) for path in files),
            "files": {
                str(path.relative_to(run_root)): {
                    "rows": line_count(path),
                    "sha256": sha256(path),
                }
                for path in files
            },
        }

    actual_input_sha256 = sha256(archive)
    if actual_input_sha256 != input_sha256:
        raise ValueError(
            "workspace archive digest mismatch "
            f"({actual_input_sha256} != {input_sha256})"
        )

    previous_status: dict[str, object] | None = None
    if status_path.exists():
        previous_status = json.loads(status_path.read_text(encoding="utf-8"))

    if resume:
        if previous_status is None:
            raise ValueError(
                f"cannot resume {run_id!r}: runtime_status.json is missing"
            )
        if previous_status.get("run_id") != run_id:
            raise ValueError("run id differs from the existing status record")
        if previous_status.get("input_sha256") != input_sha256:
            raise ValueError("input digest differs from the existing status record")
        previous_state = str(previous_status.get("state"))
        if previous_state in {"preflight", "running", "materializing"}:
            raise RuntimeError(
                f"cannot resume run with active state {previous_state!r}; inspect the "
                "existing Modal call before attempting recovery"
            )
        if previous_state == "completed":
            raise RuntimeError("completed runs must be verified, not resumed")
    elif run_root.exists():
        raise FileExistsError(
            f"run {run_id!r} already exists; choose a new run id or pass --resume"
        )

    if resume:
        if not run_root.is_dir():
            raise FileNotFoundError(run_root)
    else:
        run_root.parent.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(exist_ok=False)

    run_lock = ExclusiveFileLock(
        run_root / ".writer.lock",
        {
            "run_id": run_id,
            "input_sha256": input_sha256,
            "app_name": selected_app_name,
            "volume_name": selected_volume_name,
            "modal_function_call_id": os.environ.get("MODAL_FUNCTION_CALL_ID"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "acquired_unix": time.time(),
        },
    )
    results.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    initial_cache = cache_inventory()
    if not resume and initial_cache["rows"] != 0:
        raise RuntimeError("fresh Modal score cache is not empty")

    if not resume:
        if workspace.exists():
            shutil.rmtree(workspace)

        def workspace_only(
            member: tarfile.TarInfo, destination: str
        ) -> tarfile.TarInfo | None:
            path = PurePosixPath(member.name)
            if path.is_absolute() or not path.parts or path.parts[0] != "workspace":
                raise tarfile.FilterError(
                    f"archive member must be under workspace/: {member.name!r}"
                )
            if any(part in {"", ".", ".."} for part in path.parts):
                raise tarfile.FilterError(f"unsafe archive member: {member.name!r}")
            return tarfile.data_filter(member, destination)

        with archive.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as source:
                    source.extractall(path=run_root, filter=workspace_only)

        if not workspace.is_dir():
            raise ValueError("archive did not create the required workspace/ directory")

        # A linked-worktree .git pointer contains an absolute path from the
        # uploader's machine.  Removing VCS metadata keeps repo_cache_root()
        # anchored to this extracted workspace inside the Volume.
        git_metadata = workspace / ".git"
        if git_metadata.is_symlink() or git_metadata.is_file():
            git_metadata.unlink()
        elif git_metadata.exists():
            shutil.rmtree(git_metadata)

        # Never seed a fresh Modal run with local CPU/ROCm scores.  The scorer's
        # normal repo-relative cache path becomes a symlink to this run's empty,
        # persistent schema-v2 cache instead.
        extracted_score_cache = workspace / "cache" / "reranker" / "score_cache"
        if extracted_score_cache.is_symlink() or extracted_score_cache.is_file():
            extracted_score_cache.unlink()
        elif extracted_score_cache.exists():
            shutil.rmtree(extracted_score_cache)
        extracted_score_cache.parent.mkdir(parents=True, exist_ok=True)
        extracted_score_cache.symlink_to(cache_root, target_is_directory=True)
        marker_path.write_text(input_sha256 + "\n", encoding="utf-8")
    else:
        if (
            not marker_path.is_file()
            or marker_path.read_text(encoding="utf-8").strip() != input_sha256
        ):
            raise ValueError("resume workspace does not match input_sha256")
        linked_cache = workspace / "cache" / "reranker" / "score_cache"
        if (
            not linked_cache.is_symlink()
            or linked_cache.resolve() != cache_root.resolve()
        ):
            raise ValueError(
                "resume workspace is not linked to its run-scoped Modal cache"
            )

    required_workspace_paths = (
        workspace / "AGENTS.md",
        workspace / "code" / "trec_rag" / "rerank_score_cache.py",
        workspace / "configs" / "rag25_bm25_mixedbread_rerank_v1.yaml",
        workspace
        / "trec-rag-data"
        / "trec-rag-2026"
        / "development-data"
        / "topics"
        / "rag25-topics-dev.tsv",
        workspace / "cache" / "retrieval" / "pyserini_remote",
    )
    missing_paths = [
        str(path) for path in required_workspace_paths if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError("workspace is incomplete: " + ", ".join(missing_paths))

    versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in VERSIONED_DISTRIBUTIONS
    }
    expected_versions = {
        requirement.split("==", 1)[0]: requirement.split("==", 1)[1]
        for requirement in PINNED_PACKAGES
    }
    if versions != expected_versions:
        raise RuntimeError(
            f"installed package versions differ from the image contract: {versions!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    attempt = int(previous_status.get("attempt", 0)) + 1 if previous_status else 1
    created_unix = (
        float(previous_status.get("created_unix", time.time()))
        if previous_status
        else time.time()
    )
    scoring_contract: dict[str, object] = {
        "artifact_schema_version": 2,
        "backend": "sentence-transformers-cross-encoder",
        "backend_version": "5.6.0",
        "model": "mixedbread-ai/mxbai-rerank-base-v2",
        "model_revision": "3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
        "score_representation": "raw_logits",
        "inference_dtype": "bfloat16",
        "input_policy": "trec_rag_raw_v2",
        "candidate_limit": 1000,
        "topics": list(TOPICS),
        "document_max_length": 32768,
        "document_pair_buffer_tokens": 512,
        "document_model_max_length": 32256,
        "document_batch_size": 1,
        "window_max_length": 1024,
        "window_batch_size": 32,
        "chunk_max_characters": 3500,
        "chunk_overlap_characters": 350,
    }
    source_sha256 = {
        str(path.relative_to(workspace)): sha256(path)
        for path in required_workspace_paths[:3]
    }

    stale_lock_recovery: dict[str, object] | None = None
    if run_lock.path.exists() and recover_stale_lock:
        stale_payload = json.loads(run_lock.path.read_text(encoding="utf-8"))
        if (
            stale_payload.get("run_id") != run_id
            or stale_payload.get("input_sha256") != input_sha256
        ):
            raise RuntimeError("stale writer marker does not match this run and input")
        stale_archive_dir = run_root / "stale_locks"
        stale_archive_dir.mkdir(parents=True, exist_ok=True)
        stale_archive_path = (
            stale_archive_dir / f"writer-lock-{int(time.time() * 1_000_000)}.json"
        )
        stale_archive_path.write_text(
            json.dumps(stale_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_lock.path.unlink()
        volume.commit()
        stale_lock_recovery = {
            "recovered": True,
            "archived_marker": str(stale_archive_path.relative_to(run_root)),
            "previous_marker": stale_payload,
        }

    run_lock.acquire()
    resume_integrity: list[dict[str, object]] = []
    if resume:
        repair_root = run_root / "resume_repairs" / str(int(time.time() * 1_000_000))
        integrity_paths = [
            path for path in (document_path, window_path) if path.exists()
        ]
        schema_v2_cache_root = cache_root / "schema_v2"
        if schema_v2_cache_root.exists():
            integrity_paths.extend(sorted(schema_v2_cache_root.rglob("*.jsonl")))
        try:
            for integrity_path in integrity_paths:
                relative_path = str(integrity_path.relative_to(run_root))
                archive_bucket = hashlib.sha256(
                    relative_path.encode("utf-8")
                ).hexdigest()[:16]
                result = validate_or_repair_final_jsonl(
                    integrity_path,
                    archive_dir=repair_root / archive_bucket,
                )
                result["path"] = relative_path
                if result.get("fragment_archive"):
                    result["fragment_archive"] = str(
                        Path(str(result["fragment_archive"])).relative_to(run_root)
                    )
                resume_integrity.append(result)
        except Exception:
            run_lock.release()
            volume.commit()
            raise

    attempt_started = time.time()
    stop_heartbeat = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    status_writer = ThreadSafeJsonStatusWriter(status_path)

    def write_status(
        state: str,
        error: str | None = None,
        **extra: object,
    ) -> dict[str, object]:
        now = time.time()
        payload: dict[str, object] = {
            "state": state,
            "error": error,
            "run_id": run_id,
            "attempt": attempt,
            "resumed": resume,
            "created_unix": created_unix,
            "attempt_started_unix": attempt_started,
            "updated_unix": now,
            "attempt_elapsed_seconds": now - attempt_started,
            "app_name": selected_app_name,
            "volume_name": selected_volume_name,
            "modal_function_call_id": os.environ.get("MODAL_FUNCTION_CALL_ID"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "modal_cloud_provider": os.environ.get("MODAL_CLOUD_PROVIDER"),
            "modal_region": os.environ.get("MODAL_REGION"),
            "gpu": torch.cuda.get_device_name(0),
            "torch_cuda_version": torch.version.cuda,
            "input_archive": archive_relative_path,
            "input_sha256": input_sha256,
            "source_sha256": source_sha256,
            "versions": versions,
            "scoring_contract": scoring_contract,
            "initial_score_cache": initial_cache,
            "resume_jsonl_integrity": resume_integrity,
            "stale_lock_recovery": stale_lock_recovery,
            "document_artifact": str(document_path.relative_to(volume_root)),
            "window_artifact": str(window_path.relative_to(volume_root)),
            "document_rows": line_count(document_path),
            "window_rows": line_count(window_path),
            **extra,
        }
        status_writer.write(payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return payload

    def heartbeat() -> None:
        while not stop_heartbeat.wait(300):
            try:
                write_status("running")
                volume.commit()
            except Exception:
                traceback.print_exc()

    def stop_and_join_heartbeat() -> None:
        stop_heartbeat.set()
        if heartbeat_thread is not None and heartbeat_thread.is_alive():
            heartbeat_thread.join()

    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    python_paths = [str(workspace / "code")]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    base_command = [
        sys.executable,
        "-m",
        "trec_rag.rerank_score_cache",
        "--config",
        str(workspace / "configs" / "rag25_bm25_mixedbread_rerank_v1.yaml"),
        "--topics",
        *TOPICS,
        "--limit-per-topic",
        "1000",
        "--device",
        "cuda",
        "--sleep-between-topics",
        "0",
        "--document-score-path",
        str(document_path),
        "--window-score-path",
        str(window_path),
        "--document-batch-size",
        "1",
        "--window-batch-size",
        "32",
    ]

    def validate_artifact(
        path: Path,
        expected_counts: dict[str, int],
        *,
        window: bool,
    ) -> dict[str, object]:
        counts = {topic: 0 for topic in TOPICS}
        keys: set[tuple[object, ...]] = set()
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
                topic_id = str(row.get("topic_id"))
                if topic_id not in counts:
                    raise ValueError(
                        f"{path}:{line_number}: unexpected topic {topic_id!r}"
                    )
                expected_metadata = {
                    "artifact_schema_version": 2,
                    "backend": "sentence-transformers-cross-encoder",
                    "backend_version": "5.6.0",
                    "model": "mixedbread-ai/mxbai-rerank-base-v2",
                    "model_revision": "3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
                    "score_representation": "raw_logits",
                    "inference_dtype": "bfloat16",
                    "input_policy": "trec_rag_raw_v2",
                }
                for field, expected in expected_metadata.items():
                    if row.get(field) != expected:
                        raise ValueError(
                            f"{path}:{line_number}: {field}={row.get(field)!r}; "
                            f"expected {expected!r}"
                        )
                score = float(row["score"])
                if not math.isfinite(score):
                    raise ValueError(f"{path}:{line_number}: score is not finite")
                if not row.get("score_cache_key"):
                    raise ValueError(
                        f"{path}:{line_number}: score_cache_key is missing"
                    )
                if window:
                    key = (topic_id, str(row["docid"]), int(row["chunk_index"]))
                    if (
                        row.get("score_kind") != "window"
                        or row.get("max_length") != 1024
                    ):
                        raise ValueError(f"{path}:{line_number}: invalid window policy")
                else:
                    key = (topic_id, str(row["docid"]))
                    if (
                        row.get("score_kind") != "doc_max_32768_buf512"
                        or row.get("max_length") != 32256
                    ):
                        raise ValueError(
                            f"{path}:{line_number}: invalid document policy"
                        )
                if key in keys:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate artifact key {key!r}"
                    )
                keys.add(key)
                counts[topic_id] += 1
        if counts != expected_counts:
            raise ValueError(f"{path}: per-topic row counts differ: {counts!r}")
        return {
            "rows": sum(counts.values()),
            "rows_by_topic": counts,
            "sha256": sha256(path),
        }

    try:
        write_status("preflight")
        volume.commit()
        subprocess.run(
            [*base_command, "--dry-run"],
            cwd=workspace,
            env=environment,
            check=True,
        )
        write_status("running")
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        subprocess.run(base_command, cwd=workspace, env=environment, check=True)

        document_validation = validate_artifact(
            document_path,
            EXPECTED_DOCUMENT_ROWS_BY_TOPIC,
            window=False,
        )
        window_validation = validate_artifact(
            window_path,
            EXPECTED_WINDOW_ROWS_BY_TOPIC,
            window=True,
        )
        if document_validation["rows"] != EXPECTED_DOCUMENT_ROWS:
            raise ValueError("document artifact total row count differs")
        if window_validation["rows"] != EXPECTED_WINDOW_ROWS:
            raise ValueError("window artifact total row count differs")

        stop_and_join_heartbeat()
        final = write_status(
            "completed",
            document_validation=document_validation,
            window_validation=window_validation,
            final_score_cache=cache_inventory(),
        )
        run_lock.release()
        volume.commit()
        return final
    except Exception as exc:
        stop_and_join_heartbeat()
        write_status("failed", f"{type(exc).__name__}: {exc}")
        run_lock.release()
        volume.commit()
        raise
    finally:
        stop_and_join_heartbeat()
        if run_lock.acquired:
            run_lock.release()
            volume.commit()


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=3600,
    volumes={VOLUME_MOUNT: volume},
)
def verify_warm_cache(
    layout: str,
    verification_id: str,
    selected_app_name: str,
    selected_volume_name: str,
    run_id: str = "",
) -> dict[str, object]:
    """Rebuild artifacts on CPU using only an already-complete Modal cache."""

    from collections import Counter
    import filecmp
    import hashlib
    import json
    import math
    import os
    from pathlib import Path
    import re
    import subprocess
    import sys
    import time

    identifier_pattern = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    if layout not in {"run-scoped", "legacy"}:
        raise ValueError("layout must be 'run-scoped' or 'legacy'")
    if not re.fullmatch(identifier_pattern, verification_id):
        raise ValueError("verification_id contains unsafe characters")
    if layout == "run-scoped" and not re.fullmatch(identifier_pattern, run_id):
        raise ValueError("run-scoped verification requires a safe run_id")
    if layout == "legacy" and run_id:
        raise ValueError("legacy verification does not accept run_id")
    if not selected_app_name.strip() or not selected_volume_name.strip():
        raise ValueError("selected Modal app and Volume names must not be empty")

    volume_root = Path(VOLUME_MOUNT)
    resolved_volume_root = volume_root.resolve()

    def checked_path(*parts: str) -> Path:
        if not parts or any(
            not part or part in {".", ".."} or "/" in part or "\\" in part
            for part in parts
        ):
            raise ValueError(f"unsafe Volume path components: {parts!r}")
        candidate = volume_root.joinpath(*parts)
        if not candidate.resolve().is_relative_to(resolved_volume_root):
            raise ValueError(f"Volume path escapes its mount: {candidate}")
        return candidate

    if layout == "run-scoped":
        source_root = checked_path("runs", run_id)
        workspace = source_root / "workspace"
        canonical_results = source_root / "results"
        source_status_path = source_root / "runtime_status.json"
        canonical_document_path = canonical_results / DOCUMENT_ARTIFACT_NAME
        canonical_window_path = canonical_results / WINDOW_ARTIFACT_NAME
        source_label = f"run-{run_id}"
    else:
        workspace = checked_path("workspace")
        canonical_results = checked_path("results")
        source_status_path = canonical_results / "runtime_status.json"
        canonical_document_path = canonical_results / LEGACY_DOCUMENT_ARTIFACT_NAME
        canonical_window_path = canonical_results / LEGACY_WINDOW_ARTIFACT_NAME
        source_label = "legacy"

    verification_root = checked_path(
        "verifications",
        source_label,
        verification_id,
    )
    if verification_root.exists():
        raise FileExistsError(
            f"verification {verification_id!r} already exists for {source_label!r}; "
            "use a fresh verification id"
        )
    verification_root.mkdir(parents=True)
    regenerated_document_path = verification_root / "document_scores.jsonl"
    regenerated_window_path = verification_root / "window_scores.jsonl"
    preflight_log_path = verification_root / "preflight_stdout.log"
    scorer_log_path = verification_root / "scorer_stdout.log"
    verification_status_path = verification_root / "verification_status.json"
    started = time.time()

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def write_status(
        state: str,
        error: str | None = None,
        **extra: object,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "state": state,
            "error": error,
            "layout": layout,
            "run_id": run_id or None,
            "verification_id": verification_id,
            "app_name": selected_app_name,
            "volume_name": selected_volume_name,
            "started_unix": started,
            "updated_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "workspace": str(workspace.relative_to(volume_root)),
            "canonical_results": str(canonical_results.relative_to(volume_root)),
            "verification_root": str(verification_root.relative_to(volume_root)),
            "device": "cpu",
            "model_loading_forbidden": True,
            **extra,
        }
        temporary_path = verification_status_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(verification_status_path)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return payload

    try:
        required_paths = (
            workspace / "AGENTS.md",
            workspace / "code" / "trec_rag" / "rerank_score_cache.py",
            workspace / "configs" / "rag25_bm25_mixedbread_rerank_v1.yaml",
            workspace / "cache" / "retrieval" / "pyserini_remote",
            workspace / "cache" / "reranker" / "score_cache",
            source_status_path,
            canonical_document_path,
            canonical_window_path,
        )
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "verification source is incomplete: " + ", ".join(missing_paths)
            )

        source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
        if source_status.get("state") != "completed":
            raise RuntimeError(
                "canonical scoring run must be completed before verification; "
                f"found state={source_status.get('state')!r}"
            )

        score_cache_root = workspace / "cache" / "reranker" / "score_cache"
        score_cache_paths = sorted(score_cache_root.rglob("*.jsonl"))
        if not score_cache_paths:
            raise FileNotFoundError("Modal schema-v2 global score cache is empty")
        if any("schema_v2" not in path.parts for path in score_cache_paths):
            raise ValueError("global score cache contains a non-schema-v2 JSONL file")
        score_cache_sha256_before = {
            str(path.relative_to(volume_root)): sha256(path)
            for path in score_cache_paths
        }

        environment = dict(os.environ)
        existing_pythonpath = environment.get("PYTHONPATH")
        python_paths = [str(workspace / "code")]
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)

        scorer_arguments = [
            "--config",
            str(workspace / "configs" / "rag25_bm25_mixedbread_rerank_v1.yaml"),
            "--topics",
            *TOPICS,
            "--limit-per-topic",
            "1000",
            "--device",
            "cpu",
            "--sleep-between-topics",
            "0",
            "--document-score-path",
            str(regenerated_document_path),
            "--window-score-path",
            str(regenerated_window_path),
            "--document-batch-size",
            "1",
            "--window-batch-size",
            "32",
        ]
        preflight = subprocess.run(
            [
                sys.executable,
                "-m",
                "trec_rag.rerank_score_cache",
                *scorer_arguments,
                "--dry-run",
            ],
            cwd=workspace,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        preflight_log_path.write_text(preflight.stdout, encoding="utf-8")
        print(preflight.stdout, end="", flush=True)
        if preflight.returncode != 0:
            raise RuntimeError(f"score-cache preflight exited {preflight.returncode}")

        # If even one cache value is absent, the scorer will call this patched
        # loader and fail before model inference.  This makes the verification
        # incapable of silently filling gaps with CPU-produced scores.
        warm_cache_wrapper = """
import sys
from trec_rag import rerank_score_cache as scorer

def forbidden_model_load(*args, **kwargs):
    raise RuntimeError("WARM_CACHE_MISS: model loading is forbidden")

scorer._load_cross_encoder = forbidden_model_load
raise SystemExit(scorer.main())
"""
        write_status(
            "materializing",
            source_status_sha256=sha256(source_status_path),
            canonical_document_sha256=sha256(canonical_document_path),
            canonical_window_sha256=sha256(canonical_window_path),
        )
        materialization = subprocess.run(
            [sys.executable, "-c", warm_cache_wrapper, *scorer_arguments],
            cwd=workspace,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        scorer_log_path.write_text(materialization.stdout, encoding="utf-8")
        print(materialization.stdout, end="", flush=True)
        if materialization.returncode != 0:
            raise RuntimeError(
                "warm-cache materialization failed without permitting model inference "
                f"(exit {materialization.returncode})"
            )

        required_model_state = (
            "document_model_required=False window_model_required=False"
        )
        if required_model_state not in materialization.stdout:
            raise ValueError("scorer did not report both models as unnecessary")
        document_model_scores = [
            int(value)
            for value in re.findall(
                r"document_model_scores=(\d+)", materialization.stdout
            )
        ]
        window_model_scores = [
            int(value)
            for value in re.findall(
                r"window_model_scores=(\d+)", materialization.stdout
            )
        ]
        if (
            len(document_model_scores) != len(TOPICS)
            or len(window_model_scores) != len(TOPICS)
            or any(document_model_scores)
            or any(window_model_scores)
        ):
            raise ValueError(
                "expected one zero model-score count per topic; "
                f"document={document_model_scores!r} window={window_model_scores!r}"
            )

        cache_rows: dict[str, dict[str, object]] = {}
        cache_file_inventory: dict[str, object] = {}
        for cache_path in score_cache_paths:
            rows_in_file = 0
            with cache_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    row = json.loads(line)
                    if row.get("schema_version") != 2:
                        raise ValueError(
                            f"{cache_path}:{line_number}: expected schema_version=2"
                        )
                    cache_key = str(row["cache_key"])
                    if cache_key in cache_rows and cache_rows[cache_key] != row:
                        raise ValueError(f"conflicting global cache key {cache_key}")
                    cache_rows[cache_key] = row
                    rows_in_file += 1
            cache_file_inventory[str(cache_path.relative_to(volume_root))] = {
                "rows": rows_in_file,
                "sha256": sha256(cache_path),
            }
        score_cache_sha256_after = {
            path: str(details["sha256"])
            for path, details in cache_file_inventory.items()
        }
        if score_cache_sha256_after != score_cache_sha256_before:
            raise ValueError(
                "warm-cache verification unexpectedly modified the score cache"
            )

        def validate_regenerated_artifact(
            path: Path,
            expected_counts: dict[str, int],
            *,
            window: bool,
        ) -> dict[str, object]:
            counts: Counter[str] = Counter()
            keys: set[tuple[object, ...]] = set()
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    row = json.loads(line)
                    topic_id = str(row.get("topic_id"))
                    if topic_id not in expected_counts:
                        raise ValueError(
                            f"{path}:{line_number}: unexpected topic {topic_id!r}"
                        )
                    score = float(row["score"])
                    if not math.isfinite(score):
                        raise ValueError(f"{path}:{line_number}: non-finite score")
                    artifact_cache_key = str(row["score_cache_key"])
                    cache_row = cache_rows.get(artifact_cache_key)
                    if cache_row is None:
                        raise ValueError(
                            f"{path}:{line_number}: score_cache_key is absent from Modal cache"
                        )
                    for field, artifact_field in (
                        ("score", "score"),
                        ("query_sha256", "query_sha256"),
                        ("text_sha256", "text_sha256"),
                        ("score_kind", "score_kind"),
                        ("score_representation", "score_representation"),
                        ("inference_dtype", "inference_dtype"),
                    ):
                        if cache_row.get(field) != row.get(artifact_field):
                            raise ValueError(
                                f"{path}:{line_number}: artifact/cache {field} mismatch"
                            )
                    if window:
                        key = (topic_id, str(row["docid"]), int(row["chunk_index"]))
                    else:
                        key = (topic_id, str(row["docid"]))
                    if key in keys:
                        raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
                    keys.add(key)
                    counts[topic_id] += 1
            observed_counts = {topic: counts[topic] for topic in TOPICS}
            if observed_counts != expected_counts:
                raise ValueError(
                    f"{path}: per-topic counts differ: {observed_counts!r}"
                )
            return {
                "rows": sum(observed_counts.values()),
                "rows_by_topic": observed_counts,
                "sha256": sha256(path),
                "all_rows_matched_modal_cache": True,
            }

        regenerated_document = validate_regenerated_artifact(
            regenerated_document_path,
            EXPECTED_DOCUMENT_ROWS_BY_TOPIC,
            window=False,
        )
        regenerated_window = validate_regenerated_artifact(
            regenerated_window_path,
            EXPECTED_WINDOW_ROWS_BY_TOPIC,
            window=True,
        )

        canonical_document_sha256 = sha256(canonical_document_path)
        canonical_window_sha256 = sha256(canonical_window_path)
        document_semantic_comparison = compare_artifact_rows(
            load_jsonl_rows(str(canonical_document_path)),
            load_jsonl_rows(str(regenerated_document_path)),
            window=False,
        )
        window_semantic_comparison = compare_artifact_rows(
            load_jsonl_rows(str(canonical_window_path)),
            load_jsonl_rows(str(regenerated_window_path)),
            window=True,
        )
        document_byte_identical = regenerated_document[
            "sha256"
        ] == canonical_document_sha256 and filecmp.cmp(
            regenerated_document_path,
            canonical_document_path,
            shallow=False,
        )
        window_byte_identical = regenerated_window[
            "sha256"
        ] == canonical_window_sha256 and filecmp.cmp(
            regenerated_window_path,
            canonical_window_path,
            shallow=False,
        )

        final = write_status(
            "completed",
            source_status_sha256=sha256(source_status_path),
            score_cache_files=cache_file_inventory,
            score_cache_unique_keys=len(cache_rows),
            score_cache_unchanged=True,
            model_required={"document": False, "window": False},
            model_scores={"document": 0, "window": 0},
            regenerated_document=regenerated_document,
            regenerated_window=regenerated_window,
            canonical_document_sha256=canonical_document_sha256,
            canonical_window_sha256=canonical_window_sha256,
            document_semantic_comparison=document_semantic_comparison,
            window_semantic_comparison=window_semantic_comparison,
            semantic_equal_to_canonical=True,
            byte_identical_to_canonical={
                "document": document_byte_identical,
                "window": window_byte_identical,
            },
            cache_keys_recomputed_from_raw_text_by_scorer=True,
            posthoc_validator_recomputed_cache_keys=False,
            cache_key_note=(
                "The scorer recomputed keys from raw cached retrieval text to obtain every "
                "warm-cache hit. The posthoc validator cannot independently recompute them "
                "because artifacts retain hashes rather than raw text; it matched every "
                "artifact key, score, and query/text hash to its schema-v2 cache row."
            ),
            preflight_log_sha256=sha256(preflight_log_path),
            scorer_log_sha256=sha256(scorer_log_path),
        )
        volume.commit()
        return final
    except Exception as exc:
        failed = write_status("failed", f"{type(exc).__name__}: {exc}")
        volume.commit()
        print(json.dumps(failed, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(
    run_id: str,
    input_sha256: str,
    archive_relative_path: str = "inputs/workspace.tar.zst",
    resume: bool = False,
    recover_stale_lock: bool = False,
) -> None:
    """Spawn the long-running GPU call and print its durable identifiers."""

    call = score_all_candidates.spawn(
        run_id=run_id,
        input_sha256=input_sha256,
        selected_app_name=APP_NAME,
        selected_volume_name=VOLUME_NAME,
        archive_relative_path=archive_relative_path,
        resume=resume,
        recover_stale_lock=recover_stale_lock,
    )
    print(f"function_call_id={call.object_id}")
    print(f"volume_name={VOLUME_NAME}")
    print(f"status_path=runs/{run_id}/runtime_status.json")


@app.local_entrypoint()
def verify(
    layout: str,
    verification_id: str,
    run_id: str = "",
) -> None:
    """Spawn a CPU-only warm-cache verification in the selected Volume."""

    call = verify_warm_cache.spawn(
        layout=layout,
        verification_id=verification_id,
        selected_app_name=APP_NAME,
        selected_volume_name=VOLUME_NAME,
        run_id=run_id,
    )
    source_label = f"run-{run_id}" if layout == "run-scoped" else "legacy"
    print(f"function_call_id={call.object_id}")
    print(f"volume_name={VOLUME_NAME}")
    print(
        "status_path="
        f"verifications/{source_label}/{verification_id}/verification_status.json"
    )
