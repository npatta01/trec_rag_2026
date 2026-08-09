#!/usr/bin/env bash
set -euo pipefail

# Submit the agentic worker only from a clean, committed source snapshot.
# The plan and config are copied as private ignored transport inputs; they are
# never committed, and therefore cannot change the frozen source HEAD.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TASK_TEMPLATE="$REPO_ROOT/.dstack/rag26-agentic-retrieval-worker.yaml"
REMOTE_WRAPPER="$REPO_ROOT/code/tools/run_agentic_retrieval_worker.sh"
DSTACK_VERSION="0.20.29"
HTTPS_REMOTE_URL="https://github.com/npatta01/trec_rag_2026.git"
SCP_REMOTE_URL="git@github.com:npatta01/trec_rag_2026.git"
SSH_REMOTE_URL="ssh://git@github.com:npatta01/trec_rag_2026.git"

mode=preview
run_name=""
task_name=""
run_id=""
plan_sha256=""
artifact_prefix=""
plan_source=""
config_source=""
topic_ids=()
run_args=()
transport_tmp=""
temp_parent=""

usage() {
  cat <<'EOF'
Usage:
  apply_agentic_retrieval_worker.sh [--preview|--launch] --name NAME \
    --task-name SAFE_TASK --run-id SAFE_ID --plan-sha256 SHA256 \
    --artifact-prefix HF_PREFIX --plan PATH --config PATH --topic SAFE_ID \
    [--topic SAFE_ID]

Preview is the default and performs no live submission. --launch is explicit,
detached, and non-interactive. The default task has one topic; a two-topic
fallback is sequential and requires a separately reviewed ten-hour template.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

safe_id() { [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; }
safe_task() { [[ $1 =~ ^[a-z][a-z0-9-]{0,62}$ ]]; }
regular_file() { [[ -f $1 && ! -L $1 ]] || die "file must be regular and non-symlink: $1"; }

while (($#)); do
  case "$1" in
    --preview) mode=preview; shift ;;
    --launch) mode=launch; shift ;;
    --name) (($# >= 2)) || die "--name needs a value"; run_name=$2; shift 2 ;;
    --task-name) (($# >= 2)) || die "--task-name needs a value"; task_name=$2; shift 2 ;;
    --run-id) (($# >= 2)) || die "--run-id needs a value"; run_id=$2; shift 2 ;;
    --plan-sha256) (($# >= 2)) || die "--plan-sha256 needs a value"; plan_sha256=$2; shift 2 ;;
    --artifact-prefix) (($# >= 2)) || die "--artifact-prefix needs a value"; artifact_prefix=$2; shift 2 ;;
    --plan) (($# >= 2)) || die "--plan needs a value"; plan_source=$2; shift 2 ;;
    --config) (($# >= 2)) || die "--config needs a value"; config_source=$2; shift 2 ;;
    --topic) (($# >= 2)) || die "--topic needs a value"; topic_ids+=("$2"); shift 2 ;;
    --) shift; run_args=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# Keep the fixed-worker launcher shape usable as well: callers may put the
# worker arguments after `--` instead of spelling launcher flags twice.
while ((${#run_args[@]})); do
  case "${run_args[0]}" in
    --task-name) ((${#run_args[@]} >= 2)) || die "--task-name needs a value"; task_name=${run_args[1]}; run_args=("${run_args[@]:2}") ;;
    --run-id) ((${#run_args[@]} >= 2)) || die "--run-id needs a value"; run_id=${run_args[1]}; run_args=("${run_args[@]:2}") ;;
    --plan-sha256) ((${#run_args[@]} >= 2)) || die "--plan-sha256 needs a value"; plan_sha256=${run_args[1]}; run_args=("${run_args[@]:2}") ;;
    --artifact-prefix) ((${#run_args[@]} >= 2)) || die "--artifact-prefix needs a value"; artifact_prefix=${run_args[1]}; run_args=("${run_args[@]:2}") ;;
    --plan) ((${#run_args[@]} >= 2)) || die "--plan needs a value"; plan_source=${run_args[1]}; run_args=("${run_args[@]:2}") ;;
    --config) ((${#run_args[@]} >= 2)) || die "--config needs a value"; config_source=${run_args[1]}; run_args=("${run_args[@]:2}") ;;
    --topic) ((${#run_args[@]} >= 2)) || die "--topic needs a value"; topic_ids+=("${run_args[1]}"); run_args=("${run_args[@]:2}") ;;
    *) die "unknown worker argument: ${run_args[0]}" ;;
  esac
done

[[ $run_name =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] || die "--name must be a safe dstack run name"
safe_task "$task_name" || die "--task-name must be safe lowercase task text"
safe_id "$run_id" || die "--run-id must be a safe run ID"
[[ $plan_sha256 =~ ^[0-9a-f]{64}$ ]] || die "--plan-sha256 must be a lowercase SHA-256 digest"
[[ $artifact_prefix == hf://* && $artifact_prefix != *' '* ]] || die "--artifact-prefix must be a private hf:// prefix"
((${#topic_ids[@]} >= 1 && ${#topic_ids[@]} <= 2)) || die "one or two --topic selectors are required"
declare -A seen_topics=()
for topic_id in "${topic_ids[@]}"; do
  safe_id "$topic_id" || die "--topic must be a safe topic ID"
  [[ -z ${seen_topics[$topic_id]+present} ]] || die "--topic selectors must be unique"
  seen_topics[$topic_id]=true
done
regular_file "$plan_source"
regular_file "$config_source"
[[ $plan_source != *.env && $plan_source != *.env.local && $config_source != *.env && $config_source != *.env.local ]] || die "secret env files cannot be transported"

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
[[ -z $source_status ]] || die "source worktree must be clean before launch"
tracking_remote=$(git config --get "branch.${source_branch}.remote") || die "source branch has no tracking remote"
tracking_merge=$(git config --get "branch.${source_branch}.merge") || die "source branch has no tracking branch"
case "$tracking_merge" in refs/heads/*) tracking_branch=${tracking_merge#refs/heads/} ;; *) die "tracking ref is not a remote branch" ;; esac
remote_url=$(git remote get-url "$tracking_remote") || die "tracking remote has no URL"
case "$remote_url" in "$HTTPS_REMOTE_URL"|"$SCP_REMOTE_URL"|"$SSH_REMOTE_URL") ;; *) die "tracking remote is not the canonical repository" ;; esac

for required_path in "$TASK_TEMPLATE" "$REMOTE_WRAPPER"; do
  regular_file "$required_path"
  relative=${required_path#"$REPO_ROOT"/}
  git ls-files --error-unmatch -- "$relative" >/dev/null 2>&1 || die "required launcher input is not tracked: $relative"
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
transport_tmp=$(mktemp -d "$temp_parent/trec-rag-agentic-transport.XXXXXXXX")
trap 'if [[ -n ${transport_tmp:-} && -d $transport_tmp ]]; then rm -rf -- "$transport_tmp"; fi' EXIT
trap 'exit 130' HUP INT TERM
chmod 700 "$transport_tmp"

snapshot="$transport_tmp/repository"
git clone --quiet --no-hardlinks "$REPO_ROOT" "$snapshot"
snapshot_branch=$(git -C "$snapshot" symbolic-ref --quiet --short HEAD 2>/dev/null) || die "snapshot clone is detached"
[[ $snapshot_branch == "$source_branch" ]] || die "snapshot branch differs from source branch"
git -C "$snapshot" remote set-url origin "$remote_url"
git -C "$snapshot" update-ref "refs/remotes/origin/$tracking_branch" "$tracking_commit"
git -C "$snapshot" branch --set-upstream-to="origin/$tracking_branch" "$snapshot_branch" >/dev/null
[[ $(git -C "$snapshot" rev-parse 'HEAD^{commit}') == "$source_head" ]] || die "snapshot HEAD differs from source HEAD"
[[ $(git -C "$snapshot" rev-parse '@{upstream}^{commit}') == "$tracking_commit" ]] || die "snapshot tracking commit differs from source"
[[ ! -e $snapshot/.env && ! -e $snapshot/.env.local ]] || die "secret env file entered committed snapshot"

private_dir="$snapshot/.agentic-private"
mkdir -m 700 "$private_dir"
printf '/.agentic-private/\n' >>"$snapshot/.git/info/exclude"
cp -- "$plan_source" "$private_dir/run-plan.json"
cp -- "$config_source" "$private_dir/agentic-config.yaml"
chmod 600 "$private_dir/run-plan.json" "$private_dir/agentic-config.yaml"
[[ -z $(git -C "$snapshot" status --porcelain=v1 --untracked-files=all) ]] || die "private transport inputs were not excluded safely"

task_config="$transport_tmp/rag26-agentic-retrieval-worker.yaml"
snapshot_template="$snapshot/${TASK_TEMPLATE#"$REPO_ROOT"/}"
if ((${#topic_ids[@]} == 2)); then
  duration_mode=two-topic
else
  duration_mode=one-topic
fi
"$project_python" - "$snapshot_template" "$task_config" "$snapshot" "$duration_mode" "$private_dir" <<'PY'
from __future__ import annotations
import os, sys
from pathlib import Path
import yaml
source, destination, snapshot = map(Path, sys.argv[1:4])
duration_mode = sys.argv[4]
private_dir = Path(sys.argv[5])
value = yaml.safe_load(source.read_text(encoding="utf-8"))
if not isinstance(value, dict) or not isinstance(value.get("repos"), list) or len(value["repos"]) != 1:
    raise SystemExit("task template transport contract is invalid")
value["repos"][0]["local_path"] = str(snapshot)
value["files"] = [
    {
        "local_path": str(private_dir / "run-plan.json"),
        "path": "/dstack/run/trec_rag_2026/.agentic-private/run-plan.json",
    },
    {
        "local_path": str(private_dir / "agentic-config.yaml"),
        "path": "/dstack/run/trec_rag_2026/.agentic-private/agentic-config.yaml",
    },
]
if duration_mode == "two-topic":
    value["max_duration"] = "10h"
target = Path(destination)
target.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
os.chmod(target, 0o600)
PY

run_args=(
  --task-name "$task_name"
  --run-id "$run_id"
  --plan ".agentic-private/run-plan.json"
  --plan-sha256 "$plan_sha256"
  --config ".agentic-private/agentic-config.yaml"
  --artifact-prefix "$artifact_prefix"
)
for topic_id in "${topic_ids[@]}"; do run_args+=(--topic "$topic_id"); done

# dstack 0.20.29 resolves task templates relative to its current repository
# directory. The temporary transport directory contains the private task
# config, while all source/snapshot paths passed to dstack remain absolute.
cd "$transport_tmp"
set +e
if [[ $mode == preview ]]; then
  printf 'n\n' | "$dstack_bin" apply -f "$task_config" -n "$run_name" -- "${run_args[@]}"
  dstack_status=${PIPESTATUS[1]}
else
  "$dstack_bin" apply -f "$task_config" -n "$run_name" -y -d -- "${run_args[@]}"
  dstack_status=$?
fi
set -e
exit "$dstack_status"
