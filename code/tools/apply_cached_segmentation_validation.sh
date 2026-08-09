#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TASK_TEMPLATE="$REPO_ROOT/.dstack/rag25-cached-segmentation-validation.yaml"
REMOTE_WRAPPER="$REPO_ROOT/code/tools/run_cached_segmentation_validation.sh"
TRANSPORT_SENTINEL="/dev/null/trec-rag-cached-segmentation-requires-launcher"
DSTACK_VERSION="0.20.29"

mode=""
run_name=""
run_args=()
transport_tmp=""

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

cleanup() {
  [[ -n $transport_tmp ]] || return 0
  case "$transport_tmp" in
    /tmp/trec-rag-segmentation-transport.*) rm -rf -- "$transport_tmp" ;;
    *) printf 'ERROR: refusing unexpected cleanup path: %s\n' "$transport_tmp" >&2 ;;
  esac
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

while (($#)); do
  case "$1" in
    --preview|--launch)
      [[ -z $mode ]] || die "choose exactly one of --preview or --launch"
      mode=${1#--}
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
    *) die "unknown launcher argument: $1" ;;
  esac
done

[[ $mode == preview || $mode == launch ]] || die "choose --preview or --launch"
[[ $run_name =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] || die "--name is unsafe"
((${#run_args[@]})) || die "wrapper arguments are required after --"

cd "$REPO_ROOT"
[[ $(git rev-parse --show-toplevel) == "$REPO_ROOT" ]] || die "unexpected Git root"
source_branch=$(git symbolic-ref --quiet --short HEAD) || die "source must be on a branch"
tracking_ref=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}') \
  || die "source branch needs an upstream"
source_head=$(git rev-parse 'HEAD^{commit}')
tracking_head=$(git rev-parse "$tracking_ref^{commit}")
git merge-base --is-ancestor "$tracking_head" "$source_head" \
  || die "tracking commit is not an ancestor of HEAD"
git --no-pager diff --check "$tracking_ref"
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] \
  || die "source worktree must be clean and committed"

remote_name=$(git config --get "branch.${source_branch}.remote")
remote_url=$(git remote get-url "$remote_name")
case "$remote_url" in
  https://github.com/npatta01/trec_rag_2026.git|git@github.com:npatta01/trec_rag_2026.git|ssh://git@github.com/npatta01/trec_rag_2026.git) ;;
  *) die "tracking remote is not the configured credential-free GitHub repository" ;;
esac

for path in "$TASK_TEMPLATE" "$REMOTE_WRAPPER"; do
  [[ -f $path && ! -L $path ]] || die "required launcher input is missing: $path"
  git ls-files --error-unmatch -- "${path#"$REPO_ROOT"/}" >/dev/null \
    || die "required launcher input is not tracked"
done

project_python="$REPO_ROOT/.venv/bin/python"
[[ -x $project_python ]] || die "project Python is required"
dstack_bin=$(command -v dstack) || die "dstack is required"
[[ $($dstack_bin --version) == "$DSTACK_VERSION" ]] \
  || die "dstack $DSTACK_VERSION is required"

preflight_stdout=$(mktemp /tmp/segmentation-preflight.stdout.XXXXXXXX)
preflight_stderr=$(mktemp /tmp/segmentation-preflight.stderr.XXXXXXXX)
trap 'rm -f -- "$preflight_stdout" "$preflight_stderr"; cleanup' EXIT
if ! HF_CLI_MODE=direct bash "$REMOTE_WRAPPER" --preflight "${run_args[@]}" \
  >"$preflight_stdout" 2>"$preflight_stderr"; then
  cat "$preflight_stdout" "$preflight_stderr" >&2
  die "cached segmentation preflight failed"
fi

transport_tmp=$(mktemp -d /tmp/trec-rag-segmentation-transport.XXXXXXXX)
chmod 700 "$transport_tmp"
snapshot="$transport_tmp/repository"
git clone --quiet --no-hardlinks "$REPO_ROOT" "$snapshot"
[[ $(git -C "$snapshot" rev-parse 'HEAD^{commit}') == "$source_head" ]] \
  || die "snapshot HEAD changed"
[[ -z $(git -C "$snapshot" status --porcelain=v1 --untracked-files=all) ]] \
  || die "snapshot is not clean"
[[ ! -e $snapshot/.env && ! -e $snapshot/.env.local ]] \
  || die "a secret environment file entered transport"

task_config="$transport_tmp/rag25-cached-segmentation-validation.yaml"
"$project_python" - "$snapshot/${TASK_TEMPLATE#"$REPO_ROOT"/}" "$task_config" \
  "$snapshot" "$TRANSPORT_SENTINEL" <<'PY'
from pathlib import Path
import os
import sys
import yaml

source, destination, snapshot, sentinel = map(Path, sys.argv[1:])
value = yaml.safe_load(source.read_text())
expected = [{
    "local_path": str(sentinel),
    "path": "/dstack/run/trec_rag_2026",
    "if_exists": "error",
}]
if not isinstance(value, dict) or value.get("repos") != expected:
    raise SystemExit("task transport sentinel changed")
value["repos"][0]["local_path"] = str(snapshot)
destination.write_text(yaml.safe_dump(value, sort_keys=False))
os.chmod(destination, 0o600)
PY

set +e
cd "$transport_tmp"
if [[ $mode == preview ]]; then
  printf 'n\n' | "$dstack_bin" apply -f "$task_config" -n "$run_name" -- "${run_args[@]}"
  status=${PIPESTATUS[1]}
else
  "$dstack_bin" apply -f "$task_config" -n "$run_name" -y -d -- "${run_args[@]}"
  status=$?
fi
set -e
exit "$status"
