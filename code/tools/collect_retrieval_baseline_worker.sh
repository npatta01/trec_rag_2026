#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

publication_prefix=""
publication_dir=""
input_dir=""
input_manifest_sha256=""
source_revision=""
shared_cache=""
work_root=""
output_dir=""

usage() {
  cat <<'EOF'
Usage:
  collect_retrieval_baseline_worker.sh --publication-prefix HF_PREFIX \
    --publication-dir EMPTY_DIR --input-dir SEALED_INPUT \
    --input-manifest-sha256 SHA256 --source-revision GIT_SHA \
    --shared-cache CACHE_DIR --work-root EMPTY_DIR --output-dir EMPTY_DIR

Downloads one private manifest-last publication, verifies and replays it in a
fresh cache, transactionally merges non-conflicting scores into the shared
local cache, then performs a final shared-cache-only replay.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

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

empty_or_absent_directory() {
  local path=$1
  [[ ! -L $path ]] || return 1
  if [[ -e $path ]]; then
    [[ -d $path ]] || return 1
    [[ -z $(find "$path" -mindepth 1 -maxdepth 1 -print -quit) ]] || return 1
  fi
}

paths_overlap() {
  local first second
  first=$(realpath -m -- "$1") || return 1
  second=$(realpath -m -- "$2") || return 1
  [[ $first == "$second" || $first == "$second/"* || $second == "$first/"* ]]
}

while (($#)); do
  case "$1" in
    --publication-prefix) (($# >= 2)) || die "--publication-prefix needs a value"; publication_prefix=$2; shift 2 ;;
    --publication-dir) (($# >= 2)) || die "--publication-dir needs a value"; publication_dir=$2; shift 2 ;;
    --input-dir) (($# >= 2)) || die "--input-dir needs a value"; input_dir=$2; shift 2 ;;
    --input-manifest-sha256) (($# >= 2)) || die "--input-manifest-sha256 needs a value"; input_manifest_sha256=$2; shift 2 ;;
    --source-revision) (($# >= 2)) || die "--source-revision needs a value"; source_revision=$2; shift 2 ;;
    --shared-cache) (($# >= 2)) || die "--shared-cache needs a value"; shared_cache=$2; shift 2 ;;
    --work-root) (($# >= 2)) || die "--work-root needs a value"; work_root=$2; shift 2 ;;
    --output-dir) (($# >= 2)) || die "--output-dir needs a value"; output_dir=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

safe_hf_prefix "$publication_prefix" || die "--publication-prefix must be a safe private hf:// prefix"
[[ $input_manifest_sha256 =~ ^[0-9a-f]{64}$ ]] || die "--input-manifest-sha256 must be a lowercase SHA-256"
[[ $source_revision =~ ^[0-9a-f]{40}$ ]] || die "--source-revision must be a lowercase Git commit"
[[ -d $input_dir && ! -L $input_dir ]] || die "--input-dir must be a regular directory"
[[ ! -L $shared_cache && ( ! -e $shared_cache || -d $shared_cache ) ]] || die "--shared-cache must be a regular directory"
for destination in "$publication_dir" "$work_root" "$output_dir"; do
  empty_or_absent_directory "$destination" || die "collection destination must be absent or empty: $destination"
done
collection_paths=("$publication_dir" "$input_dir" "$shared_cache" "$work_root" "$output_dir")
for ((first_index = 0; first_index < ${#collection_paths[@]}; first_index++)); do
  for ((second_index = first_index + 1; second_index < ${#collection_paths[@]}; second_index++)); do
    paths_overlap "${collection_paths[$first_index]}" "${collection_paths[$second_index]}" \
      && die "collection paths must not overlap"
  done
done

cd "$REPO_ROOT"
venv_python="$REPO_ROOT/.venv/bin/python"
venv_hf="$REPO_ROOT/.venv/bin/hf"
[[ -x $venv_python && -x $venv_hf ]] || die "the locked project environment is required"
bucket=$(bucket_from_prefix "$publication_prefix")
bucket_receipt="${publication_dir}.bucket-info.json"
"$venv_hf" buckets info "$bucket" --format json >"$bucket_receipt"
"$venv_python" - "$bucket_receipt" "$bucket" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
if not isinstance(value, dict) or value.get("private") is not True:
    raise SystemExit(f"HF bucket is not confirmed private: {sys.argv[2]}")
PY

mkdir -p "$publication_dir"
"$venv_hf" buckets sync "$publication_prefix" "$publication_dir" --no-delete
[[ -f $publication_dir/publication-manifest.json && ! -L $publication_dir/publication-manifest.json ]] \
  || die "downloaded publication is missing publication-manifest.json"

"$venv_python" -m trec_rag.retrieval_baseline_collection \
  --publication-dir "$publication_dir" \
  --input-dir "$input_dir" \
  --input-manifest-sha256 "$input_manifest_sha256" \
  --source-revision "$source_revision" \
  --shared-cache "$shared_cache" \
  --work-root "$work_root" \
  --output-dir "$output_dir"
