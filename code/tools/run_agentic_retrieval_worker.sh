#!/usr/bin/env bash
set -euo pipefail

# Execute one or two assigned agentic topics sequentially on one GPU. Topic
# bundles are immutable: archive first, marker last, then a fresh
# download/verification round trip before the next topic starts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAX_SAFE_ID_LENGTH=128
WORK_ROOT_BASE="${AGENTIC_WORK_ROOT_BASE:-/tmp/trec-rag-agentic-workers}"

preflight=false
task_name=""
run_id=""
plan_path=""
plan_sha256=""
config_path=""
artifact_prefix=""
topic_ids=()
cache_archive=""
cache_marker=""

usage() {
  cat <<'EOF'
Usage:
  run_agentic_retrieval_worker.sh [--preflight] \
    --task-name SAFE_TASK --run-id SAFE_ID --plan PATH --plan-sha256 SHA256 \
    --config PATH --artifact-prefix HF_PREFIX --topic SAFE_ID [--topic SAFE_ID]

The default is one topic per task. A second --topic is allowed as a
capacity-scarcity fallback and is always executed sequentially on the GPU.
--preflight validates source, plan, config, and assignment inputs without
loading secrets or contacting Hugging Face. Optional --cache-archive and
--cache-marker files are published only after every topic is complete. The
launcher exposes --preview by default and --launch only as an explicit mode.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

is_safe_id() {
  [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]
}

is_safe_task_name() {
  [[ $1 =~ ^[a-z][a-z0-9-]{0,62}$ ]]
}

regular_file() {
  local path=$1
  [[ -f $path && ! -L $path ]] || die "file must be a regular, non-symlink file: $path"
}

while (($#)); do
  case "$1" in
    --preflight)
      preflight=true
      shift
      ;;
    --task-name)
      (($# >= 2)) || die "--task-name needs a value"
      task_name=$2
      shift 2
      ;;
    --run-id)
      (($# >= 2)) || die "--run-id needs a value"
      run_id=$2
      shift 2
      ;;
    --plan)
      (($# >= 2)) || die "--plan needs a value"
      plan_path=$2
      shift 2
      ;;
    --plan-sha256)
      (($# >= 2)) || die "--plan-sha256 needs a value"
      plan_sha256=$2
      shift 2
      ;;
    --config)
      (($# >= 2)) || die "--config needs a value"
      config_path=$2
      shift 2
      ;;
    --artifact-prefix)
      (($# >= 2)) || die "--artifact-prefix needs a value"
      artifact_prefix=$2
      shift 2
      ;;
    --topic)
      (($# >= 2)) || die "--topic needs a value"
      topic_ids+=("$2")
      shift 2
      ;;
    --cache-archive)
      (($# >= 2)) || die "--cache-archive needs a value"
      cache_archive=$2
      shift 2
      ;;
    --cache-marker)
      (($# >= 2)) || die "--cache-marker needs a value"
      cache_marker=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n $task_name ]] || die "--task-name is required"
[[ -n $run_id ]] || die "--run-id is required"
[[ -n $plan_path ]] || die "--plan is required"
[[ -n $plan_sha256 ]] || die "--plan-sha256 is required"
[[ -n $config_path ]] || die "--config is required"
[[ -n $artifact_prefix ]] || die "--artifact-prefix is required"
is_safe_task_name "$task_name" || die "--task-name must be safe lowercase task text"
is_safe_id "$run_id" || die "--run-id must be a safe run ID"
[[ $plan_sha256 =~ ^[0-9a-f]{64}$ ]] || die "--plan-sha256 must be a lowercase SHA-256 digest"
((${#topic_ids[@]} >= 1 && ${#topic_ids[@]} <= 2)) || die "one or two --topic selectors are required"
[[ $artifact_prefix == hf://* && $artifact_prefix != *' '* ]] || die "--artifact-prefix must be a private hf:// prefix"
artifact_path=${artifact_prefix#hf://}
[[ $artifact_path != *"/../"* && $artifact_path != */.. && $artifact_path != *"//"* ]] || die "--artifact-prefix contains an unsafe path"
if [[ -n $cache_archive && -z $cache_marker ]] || [[ -z $cache_archive && -n $cache_marker ]]; then
  die "--cache-archive and --cache-marker must be supplied together"
fi

declare -A seen_topic_ids=()
for topic_id in "${topic_ids[@]}"; do
  is_safe_id "$topic_id" || die "--topic must be a safe topic ID"
  [[ -z ${seen_topic_ids[$topic_id]+present} ]] || die "--topic selectors must be unique"
  seen_topic_ids[$topic_id]=true
  experiment_id="${run_id}-${topic_id}"
  ((${#experiment_id} <= MAX_SAFE_ID_LENGTH)) || die "generated topic experiment ID exceeds 128 characters"
done

cd "$REPO_ROOT"
repo_toplevel=$(git rev-parse --show-toplevel 2>/dev/null) || die "transported source is not a Git checkout"
[[ $repo_toplevel == "$REPO_ROOT" ]] || die "transported source resolved to an unexpected repository"
git --no-pager diff --check HEAD || die "transported source has whitespace errors"

regular_file "$plan_path"
regular_file "$config_path"
plan_path=$(cd "$(dirname "$plan_path")" && pwd -P)/$(basename "$plan_path")
config_path=$(cd "$(dirname "$config_path")" && pwd -P)/$(basename "$config_path")

plan_json_check() {
  python3 - "$plan_path" "$plan_sha256" "$run_id" "${topic_ids[@]}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

path, expected_digest, expected_run_id, *assigned = sys.argv[1:]
try:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"plan is not readable JSON: {exc}")
if not isinstance(value, dict):
    raise SystemExit("plan must be a JSON object")
if value.get("plan_sha256") != expected_digest:
    raise SystemExit("plan digest does not match --plan-sha256")
if value.get("run_id") != expected_run_id:
    raise SystemExit("plan run ID does not match --run-id")
topics_section = value.get("topics")
if isinstance(topics_section, dict) and isinstance(topics_section.get("planned"), list):
    planned = [item.get("topic_id") for item in topics_section["planned"] if isinstance(item, dict)]
else:
    planned = value.get("planned_topic_ids")
if not isinstance(planned, list) or not all(isinstance(item, str) for item in planned):
    raise SystemExit("plan has no valid planned_topic_ids")
if len(set(planned)) != len(planned):
    raise SystemExit("plan repeats a topic identity")
if any(topic not in planned for topic in assigned):
    raise SystemExit("assigned topic is absent from the installed plan")
source_section = value.get("source")
source_revision = source_section.get("revision") if isinstance(source_section, dict) else value.get("source_revision")
if source_revision is not None and (not isinstance(source_revision, str) or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None):
    raise SystemExit("plan source revision is invalid")
submodules = source_section.get("submodules", []) if isinstance(source_section, dict) else value.get("submodule_revisions", [])
if not isinstance(submodules, list):
    raise SystemExit("plan submodule revisions are invalid")
for row in submodules:
    if not isinstance(row, dict) or not isinstance(row.get("path"), str) or re.fullmatch(r"[0-9a-f]{40}", str(row.get("revision"))) is None:
        raise SystemExit("plan submodule revision is invalid")
PY
}

plan_json_check

validate_source_identity() {
  local expected_source
  expected_source=$(python3 - "$plan_path" <<'PY'
import json, sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())
source = value.get("source")
print(source.get("revision", "") if isinstance(source, dict) else value.get("source_revision", ""))
PY
  )
  actual_source=$(git rev-parse --verify HEAD) || die "source HEAD is unavailable"
  if [[ -n $expected_source && $actual_source != "$expected_source" ]]; then
    die "transported source revision differs from the authenticated plan"
  fi
  local status
  status=$(git status --porcelain=v1 --untracked-files=no --ignore-submodules=none)
  [[ -z $status ]] || die "transported tracked source or submodule state is dirty"
  local expected_submodule actual_line actual_revision actual_path
  while IFS= read -r actual_line; do
    [[ -n $actual_line ]] || continue
    [[ ${actual_line:0:1} == " " ]] || die "submodule is not at its pinned revision"
    actual_revision=${actual_line:1:40}
    actual_path=${actual_line:42}
    expected_submodule=$(python3 - "$plan_path" "$actual_path" <<'PY'
import json, sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())
source = value.get("source")
rows = source.get("submodules", []) if isinstance(source, dict) else value.get("submodule_revisions", [])
for row in rows:
    if row.get("path") == sys.argv[2]:
        print(row.get("revision", ""))
        break
PY
    )
    [[ -z $expected_submodule || $expected_submodule == "$actual_revision" ]] || die "submodule revision differs from the authenticated plan"
  done < <(git submodule status --recursive 2>/dev/null || true)
}

validate_source_identity

if $preflight; then
  printf '%s\n' \
    "preflight=ok" \
    "task_name=$task_name" \
    "run_id=$run_id" \
    "plan_sha256=$plan_sha256" \
    "topic_ids=$(IFS=,; printf '%s' "${topic_ids[*]}")" \
    "artifact_prefix=$artifact_prefix" \
    "sequential_topic_processes=1"
  exit 0
fi

# The dstack repository transport contains source, not the local virtual
# environment. Recreate the locked environment from the digest-pinned image's
# existing interpreter before any secret-dependent or live work.
git submodule update --init --recursive
tracked_status=$(git status --porcelain=v1 --untracked-files=no --ignore-submodules=none)
[[ -z $tracked_status ]] || die "transported source changed while initializing submodules"
image_python=$(command -v python3) || die "python3 is required"
case "$image_python" in
  /*) ;;
  *) die "python3 did not resolve to an absolute image path" ;;
esac
python_version=$($image_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
[[ $python_version == 3.11 || $python_version == 3.12 ]] || die "the digest-pinned image must provide Python 3.11 or 3.12 (found $python_version)"
uv sync \
  --group cuda \
  --locked \
  --no-managed-python \
  --no-python-downloads \
  --python "$image_python"

for secret_name in INDEX_URL PYSERINI_API_TOKEN OPENROUTER_API_KEY HF_TOKEN; do
  [[ -n ${!secret_name:-} ]] || die "required dstack secret is missing: $secret_name"
done

venv_python="$REPO_ROOT/.venv/bin/python"
venv_hf="$REPO_ROOT/.venv/bin/hf"
[[ -x $venv_python ]] || die "the locked project .venv is required"
[[ -x $venv_hf ]] || die "the locked project hf CLI is required"
[[ ${HF_CLI_MODE:-direct} == direct ]] || die "HF_CLI_MODE must be direct"
hf_cli() { "$venv_hf" "$@"; }

worker_root="$WORK_ROOT_BASE/$run_id/$task_name"
mkdir -p "$worker_root/config" "$worker_root/cache" "$worker_root/topics"
chmod 700 "$WORK_ROOT_BASE" "$WORK_ROOT_BASE/$run_id" "$worker_root" "$worker_root/config" "$worker_root/cache" "$worker_root/topics"
export HF_HOME="$worker_root/cache/huggingface"
export TREC_RAG_CACHE_ROOT="$worker_root/cache/retrieval"
mkdir -p "$HF_HOME" "$TREC_RAG_CACHE_ROOT"
chmod 700 "$HF_HOME" "$TREC_RAG_CACHE_ROOT"

failure_root="$REPO_ROOT/outputs/$run_id/work/failures"
mkdir -p "$failure_root"
chmod 700 "$REPO_ROOT/outputs/$run_id" "$REPO_ROOT/outputs/$run_id/work" "$failure_root"

write_failure_receipt() {
  local topic_id=$1
  local code=$2
  local status=$3
  local topic_failure_dir="$failure_root/$topic_id/$task_name"
  mkdir -p "$topic_failure_dir"
  chmod 700 "$topic_failure_dir"
  python3 - "$topic_failure_dir/failure-receipt.json" "$task_name" "$run_id" "$plan_sha256" "$topic_id" "$code" "$status" <<'PY'
from __future__ import annotations
import json, os, sys
from pathlib import Path
destination, task, run_id, digest, topic, code, status = sys.argv[1:]
payload = {
    "schema_version": "agentic_worker_failure_v1",
    "status": "failed",
    "task_name": task,
    "run_id": run_id,
    "plan_sha256": digest,
    "topic_id": topic,
    "error_code": code,
    "exit_status": int(status),
}
Path(destination).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
os.chmod(destination, 0o600)
PY
}

upload_failure_receipt() {
  local topic_id=$1
  local receipt="$failure_root/$topic_id/$task_name/failure-receipt.json"
  local remote="$artifact_prefix/failures/$topic_id/$task_name"
  local stage="$worker_root/topics/$topic_id/failure-upload"
  mkdir -p "$stage"
  cp -- "$receipt" "$stage/failure-receipt.json"
  hf_cli buckets sync "$stage" "$remote" --ignore-existing >/dev/null 2>&1 || true
}

validate_remote_listing() {
  local listing=$1
  local prefix=$2
  python3 - "$listing" "$prefix" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(value, list):
    raise SystemExit("remote listing is not a JSON array")
expected = {sys.argv[2].rstrip("/") + "/bundle.tar.zst", sys.argv[2].rstrip("/") + "/bundle-complete.json"}
actual = {row.get("path") for row in value if isinstance(row, dict) and row.get("type") == "file"}
if actual != expected:
    raise SystemExit("remote topic prefix is incomplete or contains unexpected files")
PY
}

upload_and_roundtrip() {
  local topic_id=$1
  local topic_root="$worker_root/topics/$topic_id"
  local bundle_dir="$topic_root/bundle"
  local archive="$bundle_dir/bundle.tar.zst"
  local marker="$bundle_dir/bundle-complete.json"
  local remote="$artifact_prefix/topics/$topic_id"
  local listing_before="$topic_root/listing-before.json"
  local listing_after="$topic_root/listing-after.json"
  local roundtrip="$topic_root/roundtrip"
  regular_file "$archive"
  regular_file "$marker"

  hf_cli buckets list "$remote" --recursive --format json >"$listing_before"
  python3 - "$listing_before" "$remote" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
prefix = sys.argv[2].rstrip("/") + "/"
names = {row.get("path") for row in rows if isinstance(row, dict) and row.get("type") == "file"}
allowed = {prefix + "bundle.tar.zst", prefix + "bundle-complete.json"}
if names - allowed:
    raise SystemExit("remote topic prefix contains unexpected files")
if names and names != allowed:
    raise SystemExit("remote topic prefix is a partial immutable publication")
PY

  upload_one() {
    local source=$1
    local basename=$(basename "$source")
    local stage="$topic_root/upload-$basename"
    mkdir -p "$stage"
    chmod 700 "$stage"
    cp -- "$source" "$stage/$basename"
    hf_cli buckets sync "$stage" "$remote" --ignore-existing >/dev/null
  }
  # Keep these calls visibly ordered: the marker is the eligibility signal.
  upload_one "$archive"
  upload_one "$marker"

  hf_cli buckets list "$remote" --recursive --format json >"$listing_after"
  validate_remote_listing "$listing_after" "$remote"
  rm -rf -- "$roundtrip"
  mkdir -m 700 "$roundtrip"
  hf_cli buckets cp "$remote/bundle.tar.zst" "$roundtrip/bundle.tar.zst"
  hf_cli buckets cp "$remote/bundle-complete.json" "$roundtrip/bundle-complete.json"
  cmp -s -- "$archive" "$roundtrip/bundle.tar.zst" || die "remote archive differs after round trip for topic $topic_id"
  cmp -s -- "$marker" "$roundtrip/bundle-complete.json" || die "remote completion marker differs after round trip for topic $topic_id"
  "$venv_python" -m trec_rag.agentic_retrieval_shard_bundle verify \
    "$roundtrip/bundle.tar.zst" "$roundtrip/bundle-complete.json" "$plan_sha256" >/dev/null
}

run_one_topic() (
  set -euo pipefail
  local topic_id=$1
  local topic_root="$worker_root/topics/$topic_id"
  local source_run_dir="$REPO_ROOT/outputs/$run_id/work"
  mkdir -p "$topic_root"
  chmod 700 "$topic_root"
  "$venv_python" -m trec_rag.competition_agentic_worker \
    "$config_path" --plan "$plan_path" --topic "$topic_id" >/dev/null
  mkdir -p "$topic_root/bundle"
  "$venv_python" -m trec_rag.agentic_retrieval_shard_bundle pack \
    "$source_run_dir" "$topic_id" \
    "$topic_root/bundle/bundle.tar.zst" "$topic_root/bundle/bundle-complete.json" >/dev/null
  "$venv_python" -m trec_rag.agentic_retrieval_shard_bundle verify \
    "$topic_root/bundle/bundle.tar.zst" "$topic_root/bundle/bundle-complete.json" "$plan_sha256" >/dev/null
  upload_and_roundtrip "$topic_id"
  printf 'topic_status=complete\ntopic_id=%s\n' "$topic_id"
)

active_pid=""
shutdown_status=0
shutdown_in_progress=false

group_alive() {
  [[ $1 =~ ^[1-9][0-9]*$ && $1 != 1 ]] && kill -0 -- "-$1" 2>/dev/null
}

handle_shutdown() {
  local signal_name=$1
  $shutdown_in_progress && return 0
  shutdown_in_progress=true
  case "$signal_name" in HUP) shutdown_status=129 ;; INT) shutdown_status=130 ;; TERM) shutdown_status=143 ;; esac
  if [[ -n $active_pid ]] && group_alive "$active_pid"; then
    kill -TERM -- "-$active_pid" 2>/dev/null || true
    sleep 0.2
    group_alive "$active_pid" && kill -KILL -- "-$active_pid" 2>/dev/null || true
    wait "$active_pid" 2>/dev/null || true
  fi
  printf 'ERROR: agentic worker interrupted by %s; active process group was reaped\n' "$signal_name" >&2
  exit "$shutdown_status"
}
trap 'handle_shutdown HUP' HUP
trap 'handle_shutdown INT' INT
trap 'handle_shutdown TERM' TERM
set -m

completed_topics=0
for topic_id in "${topic_ids[@]}"; do
  run_one_topic "$topic_id" &
  active_pid=$!
  if wait "$active_pid"; then
    active_pid=""
    completed_topics=$((completed_topics + 1))
  else
    topic_status=$?
    active_pid=""
    write_failure_receipt "$topic_id" "topic_execution_failed" "$topic_status"
    upload_failure_receipt "$topic_id"
    die "topic failed: $topic_id (safe receipt written)"
  fi
done

if [[ -n $cache_archive ]]; then
  regular_file "$cache_archive"
  regular_file "$cache_marker"
  cache_remote="$artifact_prefix/workers/$task_name"
  cache_stage="$worker_root/cache-upload"
  mkdir -p "$cache_stage"
  cp -- "$cache_archive" "$cache_stage/cache-bundle.tar.zst"
  cp -- "$cache_marker" "$cache_stage/cache-complete.json"
  hf_cli buckets sync "$cache_stage" "$cache_remote" --ignore-existing >/dev/null
fi

printf '%s\n' \
  "worker_status=complete" \
  "task_name=$task_name" \
  "run_id=$run_id" \
  "plan_sha256=$plan_sha256" \
  "completed_topics=$completed_topics" \
  "topic_ids=$(IFS=,; printf '%s' "${topic_ids[*]}")"
