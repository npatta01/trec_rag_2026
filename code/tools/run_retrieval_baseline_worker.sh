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

usage() {
  cat <<'EOF'
Usage:
  run_retrieval_baseline_worker.sh [--preflight] --task-name SAFE_NAME \
    --input-prefix HF_PREFIX --input-manifest-sha256 SHA256 \
    --output-prefix HF_PREFIX

The authenticated input manifest assigns exactly 119 topics. The worker scores
rag2026-1 and rag2026-18 first, verifies a fresh cache-only replay, and then
continues the other 117 topics in the same live model process.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

safe_id() { [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; }
safe_task() { [[ $1 =~ ^[a-z][a-z0-9-]{0,62}$ ]]; }
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

cd "$REPO_ROOT"
repo_toplevel=$(git rev-parse --show-toplevel 2>/dev/null) || die "worker source is not a Git checkout"
[[ $repo_toplevel == "$REPO_ROOT" ]] || die "worker source resolved to an unexpected checkout"
git --no-pager diff --check HEAD || die "worker source contains whitespace errors"

if $preflight; then
  printf '%s\n' \
    "preflight=ok" \
    "task_name=$task_name" \
    "topic_assignment=authenticated-input-manifest" \
    "required_topic_count=119" \
    "canary_topic_ids=rag2026-1,rag2026-18" \
    "input_prefix=$input_prefix" \
    "input_manifest_sha256=$input_manifest_sha256" \
    "output_prefix=$output_prefix" \
    "live_model_processes=1"
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
image_python=$("$image_python" -c 'import os,sys; print(os.path.realpath(sys.executable))') \
  || die "could not resolve the image Python interpreter"
image_python_version=$("$image_python" -c 'import platform; print(platform.python_version())') \
  || die "could not read the image Python version"
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
mkdir -p "$worker_root/input" "$worker_root/scoring" "$worker_root/publication"
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

require_archive_prefix() {
  local prefix=$1
  local receipt=$2
  "$venv_hf" buckets list "$prefix" --recursive --format json >"$receipt"
  "$venv_python" - "$receipt" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    body = stream.read()
rows = [] if not body.strip() else json.loads(body)
files = [row for row in rows if isinstance(row, dict) and row.get("type") == "file"]
if len(files) != 1 or not str(files[0].get("path", "")).endswith("/input.tar.gz"):
    raise SystemExit("input prefix must contain exactly input.tar.gz")
PY
}

verify_checksum_closure() {
  local root=$1
  "$venv_python" - "$root" <<'PY'
from hashlib import sha256
from pathlib import Path, PurePosixPath
import json, re, sys

root = Path(sys.argv[1])
sums = root / "SHA256SUMS"
manifest = root / "publication-manifest.json"
expected = set()
for line in sums.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", line)
    if match is None:
        raise SystemExit("SHA256SUMS contains an invalid row")
    relative = match.group(2)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise SystemExit("SHA256SUMS contains an unsafe path")
    path = root.joinpath(*pure.parts)
    if relative in expected or not path.is_file() or sha256(path.read_bytes()).hexdigest() != match.group(1):
        raise SystemExit("SHA256SUMS member differs")
    expected.add(relative)
actual = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit("publication contains a symlink")
    if path.is_file() and path not in {sums, manifest}:
        actual.add(path.relative_to(root).as_posix())
if actual != expected:
    raise SystemExit("publication file set differs from SHA256SUMS")
value = json.loads(manifest.read_text(encoding="utf-8"))
if value.get("status") != "complete" or value.get("sha256s_sha256") != sha256(sums.read_bytes()).hexdigest():
    raise SystemExit("publication manifest is invalid")
PY
}

input_bucket=$(bucket_from_prefix "$input_prefix")
output_bucket=$(bucket_from_prefix "$output_prefix")
require_private_bucket "$input_bucket" "$worker_root/input-bucket-info.json"
if [[ $output_bucket != "$input_bucket" ]]; then
  require_private_bucket "$output_bucket" "$worker_root/output-bucket-info.json"
fi
require_empty_prefix "$output_prefix" "$worker_root/output-listing-before.json"
require_archive_prefix "$input_prefix" "$worker_root/input-prefix-listing.json"

input_dir="$worker_root/input/private"
mkdir -m 700 "$input_dir"
input_archive="$worker_root/input/input.tar.gz"
"$venv_hf" buckets cp "$input_prefix/input.tar.gz" "$input_archive"
input_verify_receipt="$worker_root/input-verify-receipt.json"
"$venv_python" -m trec_rag.retrieval_baseline_input_bundle extract-archive \
  --archive "$input_archive" --output-dir "$input_dir" \
  --input-manifest-sha256 "$input_manifest_sha256" >"$input_verify_receipt"
"$venv_python" - "$input_verify_receipt" "$input_manifest_sha256" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
expected_topics = [f"rag2026-{index}" for index in range(119)]
if value.get("manifest_sha256") != sys.argv[2]:
    raise SystemExit("downloaded input manifest digest differs")
if value.get("topic_ids") != expected_topics:
    raise SystemExit("authenticated input is not the required 119-topic set")
if value.get("canary_topic_ids") != ["rag2026-1", "rag2026-18"]:
    raise SystemExit("authenticated input canaries differ")
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

publication="$worker_root/publication"
"$venv_python" -m trec_rag.retrieval_baseline_remote_worker \
  --input-dir "$input_dir" \
  --work-root "$worker_root/scoring" \
  --publication-dir "$publication" \
  --device cuda --batch-size 32 >"$worker_root/remote-scoring-stdout.json"
cp "$input_verify_receipt" "$publication/input-verify-receipt.json"
cp "$worker_root/remote-scoring-stdout.json" "$publication/remote-scoring-stdout.json"
"$venv_python" - "$publication/remote-scoring-receipt.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
if value.get("status") != "complete" or value.get("topic_count") != 119:
    raise SystemExit("remote scoring receipt is incomplete")
if value.get("canary_topic_ids") != ["rag2026-1", "rag2026-18"]:
    raise SystemExit("remote scoring canary receipt differs")
PY

(
  cd "$publication"
  find . -type f ! -name SHA256SUMS ! -name publication-manifest.json -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS
)
"$venv_python" - "$publication" "$task_name" "$source_revision" "$input_manifest_sha256" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
payload = {
    "input_manifest_sha256": sys.argv[4],
    "remote_scoring_receipt_sha256": sha256((root / "remote-scoring-receipt.json").read_bytes()).hexdigest(),
    "schema_version": "retrieval-baseline-publication-v1",
    "sha256s_sha256": sha256((root / "SHA256SUMS").read_bytes()).hexdigest(),
    "source_revision": sys.argv[3],
    "status": "complete",
    "task_name": sys.argv[2],
}
(root / "publication-manifest.json").write_text(
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
verify_checksum_closure "$publication"
require_private_bucket "$output_bucket" "$worker_root/output-bucket-info-before-upload.json"
require_empty_prefix "$output_prefix" "$worker_root/output-listing-before-upload.json"
"$venv_hf" buckets sync "$publication" "$output_prefix" \
  --exclude publication-manifest.json --ignore-existing
"$venv_hf" buckets cp "$publication/publication-manifest.json" \
  "$output_prefix/publication-manifest.json"

roundtrip="$worker_root/roundtrip"
mkdir -m 700 "$roundtrip"
"$venv_hf" buckets sync "$output_prefix" "$roundtrip" --no-delete
verify_checksum_closure "$roundtrip"
cmp -- "$publication/SHA256SUMS" "$roundtrip/SHA256SUMS"
cmp -- "$publication/publication-manifest.json" "$roundtrip/publication-manifest.json"
"$venv_python" -m trec_rag.retrieval_baseline_runs verify \
  --matrix-dir "$roundtrip/matrices" \
  --output-dir "$roundtrip/runs"

printf '%s\n' \
  "worker_status=complete" \
  "task_name=$task_name" \
  "required_topic_count=119" \
  "canary_topic_ids=rag2026-1,rag2026-18" \
  "output_prefix=$output_prefix"
