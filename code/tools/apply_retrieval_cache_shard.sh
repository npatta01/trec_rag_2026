#!/usr/bin/env bash
set -euo pipefail

# Couple every supported dstack preview/launch to a clean, committed-only Git
# snapshot. dstack's local repo transport otherwise includes untracked files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TASK_TEMPLATE="$REPO_ROOT/.dstack/rag26-retrieval-cache-shard.yaml"
REMOTE_WRAPPER="$REPO_ROOT/code/tools/run_retrieval_cache_shard.sh"
TRANSPORT_SENTINEL="/dev/null/trec-rag-dstack-transport-requires-launcher"
DSTACK_VERSION="0.20.29"
HTTPS_REMOTE_URL="https://github.com/npatta01/trec_rag_2026.git"
SCP_REMOTE_URL="git@github.com:npatta01/trec_rag_2026.git"
SSH_REMOTE_URL="ssh://git@github.com/npatta01/trec_rag_2026.git"

mode=""
run_name=""
run_args=()
transport_tmp=""
temp_parent=""

usage() {
  cat <<'EOF'
Usage:
  apply_retrieval_cache_shard.sh --preview --name NAME -- --topic rag2026-N --run-id SAFE_ID [--config TRACKED_PATH]
  apply_retrieval_cache_shard.sh --launch  --name NAME -- --topic rag2026-N --run-id SAFE_ID [--config TRACKED_PATH]

--preview always answers "no" to dstack's submission prompt.
--launch submits non-interactively and detached; use it only after approval.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

cleanup() {
  [[ -n $transport_tmp ]] || return 0
  case "$transport_tmp" in
    "$temp_parent"/trec-rag-dstack-transport.*)
      rm -rf -- "$transport_tmp"
      ;;
    *)
      printf 'ERROR: refusing to clean unexpected temporary path: %s\n' "$transport_tmp" >&2
      ;;
  esac
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

while (($#)); do
  case "$1" in
    --preview)
      [[ -z $mode ]] || die "choose exactly one of --preview or --launch"
      mode=preview
      shift
      ;;
    --launch)
      [[ -z $mode ]] || die "choose exactly one of --preview or --launch"
      mode=launch
      shift
      ;;
    --name)
      (($# >= 2)) || die "--name needs a value"
      run_name=$2
      shift 2
      ;;
    --)
      shift
      run_args=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown launcher argument: $1"
      ;;
  esac
done

[[ $mode == preview || $mode == launch ]] || die "choose exactly one of --preview or --launch"
[[ $run_name =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] || die "--name must be a safe dstack run name"
((${#run_args[@]})) || die "wrapper arguments are required after --"

cd "$REPO_ROOT"
repo_toplevel=$(git rev-parse --show-toplevel 2>/dev/null) || die "launcher must run from a Git checkout"
[[ $repo_toplevel == "$REPO_ROOT" ]] || die "launcher resolved an unexpected Git checkout"

source_branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null) || die "source checkout must be on a branch"
tracking_ref=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null) || die "source branch has no tracking branch"
tracking_commit=$(git rev-parse --verify "$tracking_ref^{commit}") || die "tracking branch does not resolve to a commit"
source_head=$(git rev-parse --verify 'HEAD^{commit}') || die "HEAD does not resolve to a commit"
git merge-base --is-ancestor "$tracking_commit" "$source_head" || die "tracking commit must be an ancestor of HEAD"
git --no-pager diff --check "$tracking_ref"

source_status=$(git status --porcelain=v1 --untracked-files=all)
[[ -z $source_status ]] || die "source worktree must be clean; commit intended code and remove non-ignored untracked files"

tracking_remote=$(git config --get "branch.${source_branch}.remote") || die "source branch has no tracking remote"
tracking_merge=$(git config --get "branch.${source_branch}.merge") || die "source branch has no tracking merge ref"
case "$tracking_merge" in
  refs/heads/*) tracking_branch=${tracking_merge#refs/heads/} ;;
  *) die "source tracking ref is not a remote branch" ;;
esac
remote_url=$(git remote get-url "$tracking_remote") || die "tracking remote has no URL"
case "$remote_url" in
  "$HTTPS_REMOTE_URL"|"$SCP_REMOTE_URL"|"$SSH_REMOTE_URL") ;;
  *) die "tracking remote must be the configured public GitHub repository without embedded credentials" ;;
esac

for required_path in "$TASK_TEMPLATE" "$REMOTE_WRAPPER"; do
  [[ -f $required_path && ! -L $required_path ]] || die "required tracked file is missing: $required_path"
  required_rel=${required_path#"$REPO_ROOT"/}
  git ls-files --error-unmatch -- "$required_rel" >/dev/null 2>&1 || die "required launcher input is not tracked: $required_rel"
done

project_python="$REPO_ROOT/.venv/bin/python"
project_hf="$REPO_ROOT/.venv/bin/hf"
[[ -x $project_python && -x $project_hf ]] || die "the locked project .venv with hf is required"
dstack_bin=$(command -v dstack) || die "dstack is required"
installed_dstack_version=$($dstack_bin --version 2>/dev/null) || die "could not read dstack version"
[[ $installed_dstack_version == "$DSTACK_VERSION" ]] || die "dstack $DSTACK_VERSION is required (found $installed_dstack_version)"

temp_parent=${TMPDIR:-/tmp}
[[ $temp_parent == /* && -d $temp_parent && ! -L $temp_parent ]] || die "TMPDIR must be an absolute, existing, non-symlink directory"
temp_parent=$(cd "$temp_parent" && pwd -P)
transport_tmp=$(mktemp -d "$temp_parent/trec-rag-dstack-transport.XXXXXXXX")
case "$transport_tmp" in
  "$temp_parent"/trec-rag-dstack-transport.*) ;;
  *) die "mktemp returned an unexpected path" ;;
esac
chmod 700 "$transport_tmp"

preflight_stdout="$transport_tmp/preflight.stdout"
preflight_stderr="$transport_tmp/preflight.stderr"
if ! HF_CLI_MODE=direct bash "$REMOTE_WRAPPER" --preflight "${run_args[@]}" >"$preflight_stdout" 2>"$preflight_stderr"; then
  printf 'ERROR: retrieval shard preflight failed\n' >&2
  cat "$preflight_stdout" "$preflight_stderr" >&2
  exit 2
fi

snapshot="$transport_tmp/repository"
git clone --quiet --no-hardlinks "$REPO_ROOT" "$snapshot"
snapshot_branch=$(git -C "$snapshot" symbolic-ref --quiet --short HEAD 2>/dev/null) || die "snapshot clone is detached"
[[ $snapshot_branch == "$source_branch" ]] || die "snapshot branch differs from source branch"
git -C "$snapshot" remote set-url origin "$remote_url"
git -C "$snapshot" update-ref "refs/remotes/origin/$tracking_branch" "$tracking_commit"
git -C "$snapshot" branch --set-upstream-to="origin/$tracking_branch" "$snapshot_branch" >/dev/null

[[ $(git -C "$snapshot" rev-parse 'HEAD^{commit}') == "$source_head" ]] || die "snapshot HEAD differs from source HEAD"
[[ $(git -C "$snapshot" rev-parse '@{upstream}^{commit}') == "$tracking_commit" ]] || die "snapshot tracking commit differs from source"
[[ -z $(git -C "$snapshot" status --porcelain=v1 --untracked-files=all) ]] || die "committed-only snapshot is not clean"
[[ ! -e $snapshot/.env && ! -e $snapshot/.env.local ]] || die "a secret env file entered the committed-only snapshot"

task_config="$transport_tmp/rag26-retrieval-cache-shard.yaml"
snapshot_task_template="$snapshot/${TASK_TEMPLATE#"$REPO_ROOT"/}"
[[ -f $snapshot_task_template && ! -L $snapshot_task_template ]] || die "snapshot task template is missing"
"$project_python" - "$snapshot_task_template" "$task_config" "$snapshot" "$TRANSPORT_SENTINEL" <<'PY'
from __future__ import annotations

import os
from pathlib import Path
import sys

import yaml

source, destination, snapshot, sentinel = map(Path, sys.argv[1:])
value = yaml.safe_load(source.read_text(encoding="utf-8"))
expected = [
    {
        "local_path": str(sentinel),
        "path": "/dstack/run/trec_rag_2026",
        "if_exists": "error",
    }
]
if not isinstance(value, dict) or value.get("repos") != expected:
    raise SystemExit("dstack task transport sentinel differs from the launcher contract")
value["repos"][0]["local_path"] = str(snapshot)
target = Path(destination)
target.write_text(
    yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
os.chmod(target, 0o600)
PY

dstack_status=0
set +e
cd "$transport_tmp"
if [[ $mode == preview ]]; then
  printf 'n\n' | "$dstack_bin" apply -f "$task_config" -n "$run_name" -- "${run_args[@]}"
  dstack_status=${PIPESTATUS[1]}
else
  "$dstack_bin" apply -f "$task_config" -n "$run_name" -y -d -- "${run_args[@]}"
  dstack_status=$?
fi
set -e
exit "$dstack_status"
