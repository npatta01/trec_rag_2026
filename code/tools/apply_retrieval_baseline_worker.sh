#!/usr/bin/env bash
set -euo pipefail

# Preview or submit the all-topic worker from a clean, committed source snapshot.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TASK_TEMPLATE_H200="$REPO_ROOT/.dstack/rag26-retrieval-baseline-worker.yaml"
TASK_TEMPLATE_H100="$REPO_ROOT/.dstack/rag26-retrieval-baseline-worker-h100.yaml"
REMOTE_WRAPPER="$REPO_ROOT/code/tools/run_retrieval_baseline_worker.sh"
DSTACK_VERSION=0.20.29
CANONICAL_HTTPS=https://github.com/npatta01/trec_rag_2026.git
CANONICAL_SCP=git@github.com:npatta01/trec_rag_2026.git
CANONICAL_SSH=ssh://git@github.com/npatta01/trec_rag_2026.git

mode=preview
gpu_profile=h200
run_name=""
task_name=""
input_prefix=""
input_manifest_sha256=""
output_prefix=""
transport_tmp=""

usage() {
  cat <<'EOF'
Usage:
  apply_retrieval_baseline_worker.sh [--preview|--launch] [--gpu h200|h100] \
    --name NAME --task-name SAFE_TASK --input-prefix HF_PREFIX \
    --input-manifest-sha256 SHA256 --output-prefix HF_PREFIX

Preview is the default and declines submission. Use H200 first. Select H100
only when the H200 preview has no acceptable offer. Both profiles stop at two
hours and cap hourly price at $5, so worst-case spend is at most $10.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

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
regular_file() { [[ -f $1 && ! -L $1 ]] || die "file must be regular and non-symlink: $1"; }

while (($#)); do
  case "$1" in
    --preview) mode=preview; shift ;;
    --launch) mode=launch; shift ;;
    --gpu) (($# >= 2)) || die "--gpu needs a value"; gpu_profile=$2; shift 2 ;;
    --name) (($# >= 2)) || die "--name needs a value"; run_name=$2; shift 2 ;;
    --task-name) (($# >= 2)) || die "--task-name needs a value"; task_name=$2; shift 2 ;;
    --input-prefix) (($# >= 2)) || die "--input-prefix needs a value"; input_prefix=$2; shift 2 ;;
    --input-manifest-sha256) (($# >= 2)) || die "--input-manifest-sha256 needs a value"; input_manifest_sha256=$2; shift 2 ;;
    --output-prefix) (($# >= 2)) || die "--output-prefix needs a value"; output_prefix=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$gpu_profile" in
  h200) TASK_TEMPLATE=$TASK_TEMPLATE_H200 ;;
  h100) TASK_TEMPLATE=$TASK_TEMPLATE_H100 ;;
  *) die "--gpu must be h200 or h100" ;;
esac
[[ $run_name =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] || die "--name must be a safe dstack run name"
safe_task "$task_name" || die "--task-name must be safe lowercase task text"
safe_hf_prefix "$input_prefix" || die "--input-prefix must be a safe private hf:// prefix"
safe_hf_prefix "$output_prefix" || die "--output-prefix must be a safe private hf:// prefix"
[[ $input_manifest_sha256 =~ ^[0-9a-f]{64}$ ]] \
  || die "--input-manifest-sha256 must be a lowercase SHA-256 digest"
[[ $input_prefix != "$output_prefix" ]] || die "input and output prefixes must differ"

cd "$REPO_ROOT"
repo_toplevel=$(git rev-parse --show-toplevel 2>/dev/null) || die "launcher must run from a Git checkout"
[[ $repo_toplevel == "$REPO_ROOT" ]] || die "launcher resolved an unexpected checkout"
source_branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null) || die "source checkout must be on a branch"
tracking_ref=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null) || die "source branch has no tracking branch"
tracking_commit=$(git rev-parse --verify "$tracking_ref^{commit}") || die "tracking branch does not resolve to a commit"
source_head=$(git rev-parse --verify 'HEAD^{commit}') || die "source HEAD does not resolve to a commit"
git merge-base --is-ancestor "$tracking_commit" "$source_head" || die "tracking commit must be an ancestor of HEAD"
git --no-pager diff --check "$tracking_ref"
source_status=$(git status --porcelain=v1 --untracked-files=all --ignore-submodules=none) || die "unable to inspect source worktree"
[[ -z $source_status ]] || die "source worktree must be clean before dstack preview or launch"
tracking_remote=$(git config --get "branch.${source_branch}.remote") || die "source branch has no tracking remote"
tracking_merge=$(git config --get "branch.${source_branch}.merge") || die "source branch has no tracking branch"
case "$tracking_merge" in refs/heads/*) tracking_branch=${tracking_merge#refs/heads/} ;; *) die "tracking ref is invalid" ;; esac
remote_url=$(git remote get-url "$tracking_remote") || die "tracking remote has no URL"
case "$remote_url" in "$CANONICAL_HTTPS"|"$CANONICAL_SCP"|"$CANONICAL_SSH") ;; *) die "tracking remote is not canonical" ;; esac

for required in "$TASK_TEMPLATE" "$REMOTE_WRAPPER"; do
  regular_file "$required"
  git ls-files --error-unmatch -- "${required#"$REPO_ROOT"/}" >/dev/null 2>&1 \
    || die "required launcher input is not tracked: ${required#"$REPO_ROOT"/}"
done
project_python="$REPO_ROOT/.venv/bin/python"
[[ -x $project_python ]] || die "the locked project Python is required"
dstack_bin=$(command -v dstack) || die "dstack is required"
installed_version=$($dstack_bin --version 2>/dev/null) || die "could not read dstack version"
[[ $installed_version == "$DSTACK_VERSION" ]] || die "dstack $DSTACK_VERSION is required (found $installed_version)"

temp_parent=${TMPDIR:-/tmp}
[[ $temp_parent == /* && -d $temp_parent && ! -L $temp_parent ]] || die "TMPDIR must be an absolute regular directory"
temp_parent=$(cd "$temp_parent" && pwd -P)
transport_tmp=$(mktemp -d "$temp_parent/trec-rag-baseline-transport.XXXXXXXX")
trap 'if [[ -n ${transport_tmp:-} && -d $transport_tmp ]]; then rm -rf -- "$transport_tmp"; fi' EXIT
trap 'exit 130' HUP INT TERM
chmod 700 "$transport_tmp"

snapshot="$transport_tmp/repository"
git clone --quiet --no-hardlinks "$REPO_ROOT" "$snapshot"
snapshot_branch=$(git -C "$snapshot" symbolic-ref --quiet --short HEAD 2>/dev/null) \
  || die "snapshot clone is detached"
[[ $snapshot_branch == "$source_branch" ]] || die "snapshot branch differs"
[[ $(git -C "$snapshot" rev-parse 'HEAD^{commit}') == "$source_head" ]] || die "snapshot HEAD differs"
git -C "$snapshot" remote set-url origin "$remote_url"
git -C "$snapshot" update-ref "refs/remotes/origin/$tracking_branch" "$tracking_commit"
git -C "$snapshot" branch --set-upstream-to="origin/$tracking_branch" "$snapshot_branch" >/dev/null
[[ $(git -C "$snapshot" rev-parse '@{upstream}^{commit}') == "$tracking_commit" ]] \
  || die "snapshot tracking commit differs"
[[ ! -e $snapshot/.env && ! -e $snapshot/.env.local ]] || die "secret env file entered snapshot"
task_config="$transport_tmp/$(basename "$TASK_TEMPLATE")"
"$project_python" - "$snapshot/${TASK_TEMPLATE#"$REPO_ROOT"/}" "$task_config" "$snapshot" <<'PY'
from __future__ import annotations
import os, sys
from pathlib import Path
import yaml

source, destination, snapshot = map(Path, sys.argv[1:])
value = yaml.safe_load(source.read_text(encoding="utf-8"))
if not isinstance(value, dict) or not isinstance(value.get("repos"), list) or len(value["repos"]) != 1:
    raise SystemExit("task template transport contract is invalid")
value["repos"][0]["local_path"] = str(snapshot)
destination.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
os.chmod(destination, 0o600)
PY

worker_args=(
  --task-name "$task_name"
  --input-prefix "$input_prefix"
  --input-manifest-sha256 "$input_manifest_sha256"
  --output-prefix "$output_prefix"
)
bash "$snapshot/code/tools/run_retrieval_baseline_worker.sh" --preflight "${worker_args[@]}"

cd "$transport_tmp"
set +e
if [[ $mode == preview ]]; then
  printf 'n\n' | "$dstack_bin" apply -f "$task_config" -n "$run_name" -- "${worker_args[@]}"
  dstack_status=${PIPESTATUS[1]}
else
  "$dstack_bin" apply -f "$task_config" -n "$run_name" -y -d -- "${worker_args[@]}"
  dstack_status=$?
fi
set -e
exit "$dstack_status"
