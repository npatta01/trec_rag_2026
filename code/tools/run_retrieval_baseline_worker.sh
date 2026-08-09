#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK_ROOT_BASE=/dstack/run/retrieval-baseline-work
MIXEDBREAD_MODEL=mixedbread-ai/mxbai-rerank-base-v2
MIXEDBREAD_REVISION=3ea9d4dffa7d12a4f366be8e275c349de9fc9865

preflight=false
task_name=""
input_prefix=""
input_manifest_sha256=""
output_prefix=""
topic_ids=()

usage() {
  cat <<'EOF'
Usage:
  run_retrieval_baseline_worker.sh [--preflight] --task-name SAFE_NAME \
    --input-prefix HF_PREFIX --input-manifest-sha256 SHA256 \
    --output-prefix HF_PREFIX \
    --topic SAFE_ID [--topic SAFE_ID]

The worker restores one authenticated private input directory, scores its one
or two topics sequentially on one GPU, exports the three runs, and publishes only
text-free matrices, run files, manifests, and receipts.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

safe_id() { [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; }
safe_task() { [[ $1 =~ ^[a-z][a-z0-9-]{0,62}$ ]]; }
safe_topic() { [[ $1 =~ ^rag2026-[0-9]+$ ]]; }
safe_hf_prefix() {
  local path=${1#hf://buckets/}
  local components=()
  [[ $1 == hf://buckets/* && $1 != *' '* && $path != *'//'* ]] || return 1
  IFS=/ read -r -a components <<<"$path"
  ((${#components[@]} >= 3)) || return 1
  for component in "${components[@]}"; do
    [[ $component =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && $component != . && $component != .. ]] \
      || return 1
  done
}

bucket_from_prefix() {
  local path=${1#hf://buckets/}
  local namespace=${path%%/*}
  path=${path#*/}
  local bucket=${path%%/*}
  printf 'hf://buckets/%s/%s\n' "$namespace" "$bucket"
}

while (($#)); do
  case "$1" in
    --preflight) preflight=true; shift ;;
    --task-name) (($# >= 2)) || die "--task-name needs a value"; task_name=$2; shift 2 ;;
    --input-prefix) (($# >= 2)) || die "--input-prefix needs a value"; input_prefix=$2; shift 2 ;;
    --input-manifest-sha256) (($# >= 2)) || die "--input-manifest-sha256 needs a value"; input_manifest_sha256=$2; shift 2 ;;
    --output-prefix) (($# >= 2)) || die "--output-prefix needs a value"; output_prefix=$2; shift 2 ;;
    --topic) (($# >= 2)) || die "--topic needs a value"; topic_ids+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

safe_task "$task_name" || die "--task-name must be safe lowercase task text"
safe_hf_prefix "$input_prefix" || die "--input-prefix must be a safe private hf:// prefix"
safe_hf_prefix "$output_prefix" || die "--output-prefix must be a safe private hf:// prefix"
[[ $input_manifest_sha256 =~ ^[0-9a-f]{64}$ ]] \
  || die "--input-manifest-sha256 must be a lowercase SHA-256 digest"
[[ $input_prefix != "$output_prefix" ]] || die "input and output prefixes must differ"
((${#topic_ids[@]} >= 1 && ${#topic_ids[@]} <= 2)) || die "one or two --topic selectors are required"
declare -A seen_topics=()
for topic_id in "${topic_ids[@]}"; do
  safe_topic "$topic_id" || die "--topic must be an official topic ID"
  [[ -z ${seen_topics[$topic_id]+present} ]] || die "--topic selectors must be unique"
  seen_topics[$topic_id]=true
done

cd "$REPO_ROOT"
repo_toplevel=$(git rev-parse --show-toplevel 2>/dev/null) || die "worker source is not a Git checkout"
[[ $repo_toplevel == "$REPO_ROOT" ]] || die "worker source resolved to an unexpected checkout"
git --no-pager diff --check HEAD || die "worker source contains whitespace errors"

if $preflight; then
  printf '%s\n' \
    "preflight=ok" \
    "task_name=$task_name" \
    "topic_ids=$(IFS=,; printf '%s' "${topic_ids[*]}")" \
    "input_prefix=$input_prefix" \
    "input_manifest_sha256=$input_manifest_sha256" \
    "output_prefix=$output_prefix" \
    "sequential_topic_processes=1"
  exit 0
fi

[[ -n ${HF_TOKEN:-} ]] || die "required dstack secret is missing: HF_TOKEN"
[[ ${HF_CLI_MODE:-direct} == direct ]] || die "HF_CLI_MODE must be direct"
git submodule update --init --recursive
git_status=$(git status --porcelain=v1 --untracked-files=no --ignore-submodules=none)
[[ -z $git_status ]] || die "tracked worker source or submodule state is dirty"
source_revision=$(git rev-parse 'HEAD^{commit}')
export TREC_RAG_SOURCE_REVISION="$source_revision"
image_python=$(command -v python3) || die "the pinned image has no python3 interpreter"
image_python=$(
  "$image_python" -c 'import os,sys; print(os.path.realpath(sys.executable))'
) || die "could not resolve the image Python interpreter"
image_python_version=$(
  "$image_python" -c 'import platform; print(platform.python_version())'
) || die "could not read the image Python version"
case "$image_python_version" in
  3.11.*|3.12.*) ;;
  *) die "image Python must be 3.11 or 3.12 (found $image_python_version)" ;;
esac
uv sync --group cuda --locked \
  --no-managed-python --no-python-downloads --python "$image_python"
venv_python="$REPO_ROOT/.venv/bin/python"
venv_hf="$REPO_ROOT/.venv/bin/hf"
[[ -x $venv_python && -x $venv_hf ]] || die "locked CUDA environment is incomplete"

worker_root="$WORK_ROOT_BASE/$task_name"
[[ ! -e $worker_root ]] || die "worker task directory already exists"
mkdir -p "$worker_root/input" "$worker_root/cache" "$worker_root/outputs" \
  "$worker_root/matrices" "$worker_root/runs" "$worker_root/publication"
chmod 700 "$WORK_ROOT_BASE" "$worker_root" "$worker_root"/*
export HF_HOME="$worker_root/huggingface"
mkdir -m 700 "$HF_HOME"

require_private_bucket() {
  local bucket=$1
  local receipt=$2
  "$venv_hf" buckets info "$bucket" --format json >"$receipt"
  "$venv_python" - "$receipt" "$bucket" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
if not isinstance(value, dict) or value.get("private") is not True:
    raise SystemExit(f"HF bucket is not confirmed private: {sys.argv[2]}")
PY
}

require_empty_prefix() {
  local prefix=$1
  local receipt=$2
  "$venv_hf" buckets list "$prefix" --recursive --format json >"$receipt"
  "$venv_python" - "$receipt" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    body = stream.read()
rows = [] if not body.strip() else json.loads(body)
if not isinstance(rows, list):
    raise SystemExit("output prefix listing is invalid")
if any(isinstance(row, dict) and row.get("type") == "file" for row in rows):
    raise SystemExit("output prefix is not empty")
PY
}

verify_checksum_closure() {
  local root=$1
  "$venv_python" - "$root" <<'PY'
from pathlib import Path, PurePosixPath
import re, sys

root = Path(sys.argv[1])
sums = root / "SHA256SUMS"
expected = set()
for line in sums.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", line)
    if match is None:
        raise SystemExit("SHA256SUMS contains an invalid row")
    relative = match.group(2)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise SystemExit("SHA256SUMS contains an unsafe path")
    if relative in expected:
        raise SystemExit("SHA256SUMS repeats a path")
    expected.add(relative)
actual = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit("publication contains a symlink")
    if path.is_file() and path != sums:
        actual.add(path.relative_to(root).as_posix())
if actual != expected:
    raise SystemExit("publication file set differs from SHA256SUMS")
PY
}

input_bucket=$(bucket_from_prefix "$input_prefix")
output_bucket=$(bucket_from_prefix "$output_prefix")
require_private_bucket "$input_bucket" "$worker_root/input-bucket-info.json"
if [[ $output_bucket != "$input_bucket" ]]; then
  require_private_bucket "$output_bucket" "$worker_root/output-bucket-info.json"
fi
require_empty_prefix "$output_prefix" "$worker_root/output-listing-before.json"

input_dir="$worker_root/input/private"
mkdir -m 700 "$input_dir"
"$venv_hf" buckets sync "$input_prefix" "$input_dir" --no-delete
verify_args=()
for topic_id in "${topic_ids[@]}"; do verify_args+=(--topic "$topic_id"); done
input_verify_receipt="$worker_root/input-verify-receipt.json"
"$venv_python" -m trec_rag.retrieval_baseline_input_bundle verify \
  "$input_dir" "${verify_args[@]}" >"$input_verify_receipt"
source_run_id=$("$venv_python" - "$input_verify_receipt" "$input_manifest_sha256" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
if value.get("manifest_sha256") != sys.argv[2]:
    raise SystemExit("downloaded input manifest digest differs from approved digest")
print(value["source_run_id"])
PY
)
safe_id "$source_run_id" || die "input source run ID is unsafe"
"$venv_python" -m trec_rag.retrieval_baseline_input_bundle import-scores \
  "$input_dir" --score-cache "$worker_root/cache/reranker" \
  >"$worker_root/input-score-import-receipt.json"
"$venv_python" - "$input_verify_receipt" "$worker_root/input-score-import-receipt.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    source = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    imported = json.load(stream)
expected = source["cache_stats"]["hits"]
if imported.get("source_row_count") != expected or imported.get("inserted_count") != expected:
    raise SystemExit("sealed input cache rows were not imported exactly")
PY

"$venv_python" - "$MIXEDBREAD_MODEL" "$MIXEDBREAD_REVISION" <<'PY'
from __future__ import annotations

import sys
import torch
from huggingface_hub import snapshot_download

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
model, revision = sys.argv[1:]
snapshot_download(model, revision=revision)
snapshot_download(model, revision=revision, local_files_only=True)
print(f"cuda_device={torch.cuda.get_device_name(0)}")
print(f"model_snapshot_verified={model}@{revision}")
PY

for topic_id in "${topic_ids[@]}"; do
  mkdir -p "$worker_root/matrices/$topic_id"
  "$venv_python" -m trec_rag.retrieval_baseline_runs score-topic \
    --source-dir "$input_dir/source/$source_run_id" \
    --document-store "$input_dir/documents/v1" \
    --score-cache "$worker_root/cache/reranker" \
    --output-dir "$worker_root/matrices" \
    --topic "$topic_id" \
    --device cuda \
    --batch-size 32 \
    >"$worker_root/matrices/$topic_id/score-receipt.json"
done

publication="$worker_root/publication"
mkdir -p "$publication/portable-scores" "$publication/replay-receipts"
complete_scores="$publication/portable-scores/complete.jsonl"
cache_export_receipt="$publication/cache-export-receipt.json"
"$venv_python" -m trec_rag.retrieval_baseline_input_bundle export-cache \
  --score-cache "$worker_root/cache/reranker" --output "$complete_scores" \
  >"$cache_export_receipt"
"$venv_python" - "$input_verify_receipt" "$cache_export_receipt" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    source = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    exported = json.load(stream)
expected = source["cache_stats"]["hits"] + source["cache_stats"]["misses"]
if exported.get("row_count") != expected:
    raise SystemExit("complete cache export row count differs from sealed workload")
PY

replay_cache="$worker_root/replay-cache"
"$venv_python" -m trec_rag.retrieval_baseline_input_bundle import-cache \
  --portable "$complete_scores" --score-cache "$replay_cache" \
  >"$publication/cache-replay-import-receipt.json"
for topic_id in "${topic_ids[@]}"; do
  replay_topic="$worker_root/replay-matrices/$topic_id"
  mkdir -p "$replay_topic"
  replay_receipt="$publication/replay-receipts/$topic_id.json"
  "$venv_python" -m trec_rag.retrieval_baseline_runs score-topic \
    --source-dir "$input_dir/source/$source_run_id" \
    --document-store "$input_dir/documents/v1" \
    --score-cache "$replay_cache" \
    --output-dir "$worker_root/replay-matrices" \
    --topic "$topic_id" \
    --device cuda \
    --batch-size 32 \
    --cache-only \
    >"$replay_receipt"
  "$venv_python" - "$replay_receipt" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
stats = value.get("cache_stats")
if (
    not isinstance(stats, dict)
    or stats.get("cache_hits", 0) <= 0
    or stats.get("cache_misses") != 0
    or stats.get("model_batches") != 0
):
    raise SystemExit("fresh-cache replay was not completely cache-only")
PY
  cmp -- "$worker_root/matrices/$topic_id/topic-matrix.jsonl" \
    "$replay_topic/topic-matrix.jsonl"
  cmp -- "$worker_root/matrices/$topic_id/topic-matrix-manifest.json" \
    "$replay_topic/topic-matrix-manifest.json"
done

"$venv_python" -m trec_rag.retrieval_baseline_runs rank \
  --matrix-dir "$worker_root/matrices" \
  --output-dir "$worker_root/runs" \
  >"$worker_root/runs/rank-receipt.json"
"$venv_python" -m trec_rag.retrieval_baseline_runs verify \
  --matrix-dir "$worker_root/matrices" \
  --output-dir "$worker_root/runs" \
  >"$worker_root/runs/verify-receipt.json"

mv "$worker_root/matrices" "$publication/matrices"
mv "$worker_root/runs" "$publication/runs"
"$venv_python" - "$publication/worker-receipt.json" "$task_name" \
  "$source_revision" "$input_manifest_sha256" "$cache_export_receipt" \
  "${topic_ids[@]}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

destination, task_name, source_revision, input_manifest_sha256, cache_receipt_path, *topic_ids = sys.argv[1:]
with open(cache_receipt_path, encoding="utf-8") as stream:
    cache_receipt = json.load(stream)
payload = {
    "schema_version": "retrieval-baseline-worker-receipt-v1",
    "source_revision": source_revision,
    "input_manifest_sha256": input_manifest_sha256,
    "portable_cache_row_count": cache_receipt["row_count"],
    "portable_cache_sha256": cache_receipt["sha256"],
    "status": "complete",
    "task_name": task_name,
    "topic_ids": topic_ids,
}
Path(destination).write_text(
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
(
  cd "$publication"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS
)
verify_checksum_closure "$publication"
require_private_bucket "$output_bucket" "$worker_root/output-bucket-info-before-upload.json"
require_empty_prefix "$output_prefix" "$worker_root/output-listing-before-upload.json"
"$venv_hf" buckets sync "$publication" "$output_prefix" --ignore-existing

roundtrip="$worker_root/roundtrip"
mkdir -m 700 "$roundtrip"
"$venv_hf" buckets sync "$output_prefix" "$roundtrip" --no-delete
(
  cd "$roundtrip"
  sha256sum -c SHA256SUMS
)
cmp -- "$publication/SHA256SUMS" "$roundtrip/SHA256SUMS"
verify_checksum_closure "$roundtrip"
"$venv_python" -m trec_rag.retrieval_baseline_runs verify \
  --matrix-dir "$roundtrip/matrices" \
  --output-dir "$roundtrip/runs"

printf '%s\n' \
  "worker_status=complete" \
  "task_name=$task_name" \
  "topic_ids=$(IFS=,; printf '%s' "${topic_ids[*]}")" \
  "output_prefix=$output_prefix"
