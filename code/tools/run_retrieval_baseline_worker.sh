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
output_prefix=""
topic_ids=()

usage() {
  cat <<'EOF'
Usage:
  run_retrieval_baseline_worker.sh [--preflight] --task-name SAFE_NAME \
    --input-prefix HF_PREFIX --output-prefix HF_PREFIX \
    --topic SAFE_ID [--topic SAFE_ID]

The worker restores one or two authenticated input cache bundles, scores each
topic sequentially on one GPU, exports the three runs, and publishes only
text-free matrices, run files, manifests, and receipts.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

safe_id() { [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; }
safe_task() { [[ $1 =~ ^[a-z][a-z0-9-]{0,62}$ ]]; }
safe_hf_prefix() {
  local path=${1#hf://}
  [[ $1 == hf://buckets/* && $1 != *' '* && $path != *'//'* && $path != *'/../'* && $path != */.. ]]
}

while (($#)); do
  case "$1" in
    --preflight) preflight=true; shift ;;
    --task-name) (($# >= 2)) || die "--task-name needs a value"; task_name=$2; shift 2 ;;
    --input-prefix) (($# >= 2)) || die "--input-prefix needs a value"; input_prefix=$2; shift 2 ;;
    --output-prefix) (($# >= 2)) || die "--output-prefix needs a value"; output_prefix=$2; shift 2 ;;
    --topic) (($# >= 2)) || die "--topic needs a value"; topic_ids+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

safe_task "$task_name" || die "--task-name must be safe lowercase task text"
safe_hf_prefix "$input_prefix" || die "--input-prefix must be a safe private hf:// prefix"
safe_hf_prefix "$output_prefix" || die "--output-prefix must be a safe private hf:// prefix"
[[ $input_prefix != "$output_prefix" ]] || die "input and output prefixes must differ"
((${#topic_ids[@]} >= 1 && ${#topic_ids[@]} <= 2)) || die "one or two --topic selectors are required"
declare -A seen_topics=()
for topic_id in "${topic_ids[@]}"; do
  safe_id "$topic_id" || die "--topic must be a safe topic ID"
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
    "output_prefix=$output_prefix" \
    "sequential_topic_processes=1"
  exit 0
fi

[[ -n ${HF_TOKEN:-} ]] || die "required dstack secret is missing: HF_TOKEN"
[[ ${HF_CLI_MODE:-direct} == direct ]] || die "HF_CLI_MODE must be direct"
required_python=$(<.python-version)
[[ $required_python =~ ^3\.12\.[0-9]+$ ]] || die ".python-version must pin Python 3.12"
git submodule update --init --recursive
git_status=$(git status --porcelain=v1 --untracked-files=no --ignore-submodules=none)
[[ -z $git_status ]] || die "tracked worker source or submodule state is dirty"
source_revision=$(git rev-parse 'HEAD^{commit}')
export TREC_RAG_SOURCE_REVISION="$source_revision"
uv sync --group cuda --locked --python "$required_python"
venv_python="$REPO_ROOT/.venv/bin/python"
venv_hf="$REPO_ROOT/.venv/bin/hf"
[[ -x $venv_python && -x $venv_hf ]] || die "locked CUDA environment is incomplete"

worker_root="$WORK_ROOT_BASE/$task_name"
mkdir -p "$worker_root/input" "$worker_root/cache" "$worker_root/outputs" \
  "$worker_root/matrices" "$worker_root/runs" "$worker_root/publication"
chmod 700 "$WORK_ROOT_BASE" "$worker_root" "$worker_root"/*
export HF_HOME="$worker_root/huggingface"
mkdir -m 700 "$HF_HOME"

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
  bundle_dir="$worker_root/input/$topic_id"
  mkdir -m 700 "$bundle_dir"
  "$venv_hf" buckets sync "$input_prefix/$topic_id" "$bundle_dir" --no-delete
  "$venv_python" -m trec_rag.competition_cache_bundle verify "$bundle_dir"
done

bundle_args=()
for topic_id in "${topic_ids[@]}"; do
  bundle_args+=("$worker_root/input/$topic_id")
done
"$venv_python" -m trec_rag.competition_cache_bundle merge \
  --cache-root "$worker_root/cache" \
  --outputs-root "$worker_root/outputs" \
  --score-conflicts strict \
  "${bundle_args[@]}"

for topic_id in "${topic_ids[@]}"; do
  mkdir -p "$worker_root/matrices/$topic_id"
  "$venv_python" -m trec_rag.retrieval_baseline_runs score-topic \
    --source-dir "$worker_root/outputs/facet-deepseek-b40-v3" \
    --document-store "$worker_root/cache/documents/v1" \
    --score-cache "$worker_root/cache/reranker" \
    --output-dir "$worker_root/matrices" \
    --topic "$topic_id" \
    --device cuda \
    --batch-size 32 \
    >"$worker_root/matrices/$topic_id/score-receipt.json"
done

"$venv_python" -m trec_rag.retrieval_baseline_runs rank \
  --matrix-dir "$worker_root/matrices" \
  --output-dir "$worker_root/runs" \
  >"$worker_root/runs/rank-receipt.json"
"$venv_python" -m trec_rag.retrieval_baseline_runs verify \
  --matrix-dir "$worker_root/matrices" \
  --output-dir "$worker_root/runs" \
  >"$worker_root/runs/verify-receipt.json"

publication="$worker_root/publication"
mv "$worker_root/matrices" "$publication/matrices"
mv "$worker_root/runs" "$publication/runs"
"$venv_python" - "$publication/worker-receipt.json" "$task_name" "$source_revision" "${topic_ids[@]}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

destination, task_name, source_revision, *topic_ids = sys.argv[1:]
payload = {
    "schema_version": "retrieval-baseline-worker-receipt-v1",
    "source_revision": source_revision,
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

listing="$worker_root/output-listing.json"
"$venv_hf" buckets list "$output_prefix" --recursive --format json >"$listing"
"$venv_python" - "$listing" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(rows, list):
    raise SystemExit("output prefix listing is invalid")
if any(isinstance(row, dict) and row.get("type") == "file" for row in rows):
    raise SystemExit("output prefix is not empty")
PY
"$venv_hf" buckets sync "$publication" "$output_prefix" --ignore-existing

roundtrip="$worker_root/roundtrip"
mkdir -m 700 "$roundtrip"
"$venv_hf" buckets sync "$output_prefix" "$roundtrip" --no-delete
(
  cd "$roundtrip"
  sha256sum -c SHA256SUMS
)
"$venv_python" -m trec_rag.retrieval_baseline_runs verify \
  --matrix-dir "$roundtrip/matrices" \
  --output-dir "$roundtrip/runs"

printf '%s\n' \
  "worker_status=complete" \
  "task_name=$task_name" \
  "topic_ids=$(IFS=,; printf '%s' "${topic_ids[*]}")" \
  "output_prefix=$output_prefix"
