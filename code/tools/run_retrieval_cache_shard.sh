#!/usr/bin/env bash
set -euo pipefail

# Run one or two non-agentic competition-retrieval topics on an ephemeral CUDA
# host, publish each immutable cache bundle to its private Hugging Face Bucket
# prefix, then download and verify the published bytes before reporting success.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUCKET_ID="Npatta01/trec_mlm_2026"
BUCKET_REPO_PREFIX="trec_rag_2026"
DEFAULT_CONFIG="configs/rag26_competition_retrieval_v2.yaml"
MIXEDBREAD_REVISION="3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
MINILM_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

preflight=false
topic_ids=()
run_id=""
config_arg="$DEFAULT_CONFIG"

usage() {
  cat <<'EOF'
Usage:
  run_retrieval_cache_shard.sh [--preflight] --topic SAFE_ID [--topic SAFE_ID] --run-id SAFE_ID [--config PATH]

--topic accepts a safe topic ID that must be present in the configured topics.
Examples: numeric RAG25 topic 31; RAG26 topic rag2026-0.

--preflight validates the real wrapper's tools and nested argv without needing
credentials, installing dependencies, downloading models, running retrieval,
or contacting Hugging Face.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --preflight)
      preflight=true
      shift
      ;;
    --topic)
      (($# >= 2)) || die "--topic needs a value"
      topic_ids+=("$2")
      shift 2
      ;;
    --run-id)
      (($# >= 2)) || die "--run-id needs a value"
      run_id=$2
      shift 2
      ;;
    --config)
      (($# >= 2)) || die "--config needs a value"
      config_arg=$2
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

((${#topic_ids[@]} >= 1 && ${#topic_ids[@]} <= 2)) \
  || die "one or two --topic selectors are required"
declare -A seen_topic_ids=()
for selected_topic_id in "${topic_ids[@]}"; do
  [[ $selected_topic_id =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]] \
    || die "--topic must be a safe configured topic ID"
  [[ -z ${seen_topic_ids[$selected_topic_id]+present} ]] \
    || die "--topic selectors must be unique"
  seen_topic_ids[$selected_topic_id]=true
done
[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || die "--run-id must be a safe run ID"
topic_ids_csv=$(IFS=,; printf '%s' "${topic_ids[*]}")

case "$config_arg" in
  /*) die "--config must name a tracked, repository-relative config" ;;
esac

cd "$REPO_ROOT"
repo_toplevel=$(git rev-parse --show-toplevel 2>/dev/null) || die "the dstack repo transport did not provide a Git checkout"
[[ $repo_toplevel == "$REPO_ROOT" ]] || die "wrapper must run from its transported repository"
if tracking_ref=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null); then
  git --no-pager diff --check "$tracking_ref"
else
  # Dstack may preserve the committed checkout bytes while omitting local
  # branch-upstream metadata. The launcher already validates the real source
  # upstream and seals a clean committed-only snapshot before submission.
  git --no-pager diff --check HEAD
fi

# Dstack materializes files added since the tracking commit as untracked patch
# bytes. Seal the complete sanitized patch before using Git's tracked-file
# index to validate --config. Local --preflight never mutates.
if ! $preflight; then
  git config user.name >/dev/null 2>&1 || git config user.name "dstack cache shard"
  git config user.email >/dev/null 2>&1 || git config user.email "dstack-cache-shard@invalid.local"
  git add -A
  git --no-pager diff --cached --check
  if ! git diff --cached --quiet; then
    git commit --no-gpg-sign -m "dstack ephemeral cache shard ${run_id} ${topic_ids_csv}" >/dev/null
  fi
  tracked_status=$(git status --porcelain=v1 --untracked-files=all)
  [[ -z $tracked_status ]] || die "transported checkout is not clean after the ephemeral commit"
fi

source_config_rel=$(git ls-files --full-name -- "$config_arg")
[[ -n $source_config_rel && $source_config_rel != *$'\n'* ]] || die "--config must name a tracked, repository-relative config"
source_config="$REPO_ROOT/$source_config_rel"
[[ -f $source_config && ! -L $source_config ]] || die "config must be a regular, non-symlink file: $config_arg"

for command_name in bash git python3 uv; do
  command -v "$command_name" >/dev/null 2>&1 || die "$command_name is required"
done

hf_mode=${HF_CLI_MODE:-direct}
[[ $hf_mode == direct ]] || die "HF_CLI_MODE must be direct"
venv_hf="$REPO_ROOT/.venv/bin/hf"
hf_cli() {
  "$venv_hf" "$@"
}
if $preflight; then
  [[ -x $venv_hf ]] || die "the locked project environment does not provide hf"
fi

validate_configured_topic() {
  local interpreter=$1
  local topic_id=$2
  "$interpreter" - "$source_config" "$topic_id" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys

from trec_rag.facet_pilot_config import load_facet_pilot_config, select_configured_topics

source, topic_id = sys.argv[1:]
loaded = load_facet_pilot_config(Path(source))
try:
    selected = select_configured_topics(loaded, topic_ids=(topic_id,))
except ValueError as exc:
    raise SystemExit(
        f"topic {topic_id!r} is absent from the configured topics: {exc}"
    ) from exc
if len(selected) != 1 or selected[0].id != topic_id:
    raise SystemExit(f"topic {topic_id!r} is absent from the configured topics")
PY
}

if $preflight; then
  untracked_files=()
  while IFS= read -r -d '' untracked_file; do
    untracked_files+=("$untracked_file")
  done < <(git ls-files --others --exclude-standard -z)
  if ((${#untracked_files[@]})); then
    printf 'ERROR: non-ignored untracked files would enter dstack repo transport:\n' >&2
    printf '  %s\n' "${untracked_files[@]}" >&2
    exit 2
  fi
  preflight_python="$REPO_ROOT/.venv/bin/python"
  [[ -x $preflight_python ]] || die "project .venv is required for configured-topic preflight"
  for selected_topic_id in "${topic_ids[@]}"; do
    validate_configured_topic "$preflight_python" "$selected_topic_id"
  done
  printf '%s\n' \
    "preflight=ok" \
    "topic_ids=$topic_ids_csv" \
    "parallel_topic_processes=${#topic_ids[@]}"
  for selected_topic_id in "${topic_ids[@]}"; do
    selected_experiment_id="${run_id}-${selected_topic_id}"
    selected_config_rel="configs/local/${selected_experiment_id}.yaml"
    selected_work_root="/tmp/trec-rag-cache-shards/${run_id}/${selected_topic_id}"
    selected_cache_root="$selected_work_root/cache"
    selected_bundle_dir="$selected_work_root/bundle"
    selected_remote_prefix="hf://buckets/${BUCKET_ID}/${BUCKET_REPO_PREFIX}/experiments/${run_id}/${selected_topic_id}"
    printf '%s\n' \
      "topic_id=$selected_topic_id" \
      "experiment_id=$selected_experiment_id" \
      "cache_root=$selected_cache_root" \
      "output_root=$REPO_ROOT/outputs/$selected_experiment_id" \
      "remote_prefix=$selected_remote_prefix" \
      "runner_argv=.venv/bin/python -m trec_rag.competition_retrieval $selected_config_rel --topic $selected_topic_id" \
      "pack_argv=.venv/bin/python -m trec_rag.competition_cache_bundle pack --config $selected_config_rel --topic $selected_topic_id --destination $selected_bundle_dir"
  done
  exit 0
fi

for secret_name in HF_TOKEN INDEX_URL PYSERINI_API_TOKEN OPENROUTER_API_KEY; do
  [[ -n ${!secret_name:-} ]] || die "required dstack secret is missing: $secret_name"
done

git submodule update --init --recursive
tracked_status=$(git status --porcelain=v1 --untracked-files=all)
[[ -z $tracked_status ]] || die "transported checkout changed while initializing submodules"

image_python=$(command -v python3)
case "$image_python" in
  /*) ;;
  *) die "python3 did not resolve to an absolute image path" ;;
esac
python_version=$($image_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
[[ $python_version == 3.11 || $python_version == 3.12 ]] || die "the digest-pinned image must provide Python 3.11 or 3.12 (found $python_version)"

# Bind uv to the interpreter already present in the digest-pinned image. These
# switches prohibit a hidden managed-Python download or interpreter change.
uv sync \
  --group cuda \
  --locked \
  --no-managed-python \
  --no-python-downloads \
  --python "$image_python"
venv_python="$REPO_ROOT/.venv/bin/python"
[[ -x $venv_python ]] || die "uv did not create the expected project interpreter"
venv_hf="$REPO_ROOT/.venv/bin/hf"
[[ -x $venv_hf ]] || die "the locked project environment did not install hf"

run_root="/tmp/trec-rag-cache-shards/${run_id}"
shared_root="$run_root/shared"
config_root="$REPO_ROOT/configs/local"
export HF_HOME="$shared_root/huggingface"
mkdir -p "$run_root" "$shared_root" "$HF_HOME" "$config_root"
chmod 700 "$run_root" "$shared_root" "$HF_HOME"

# Validate every selector and generate independent one-worker configs before
# downloading models or making any hosted request.
for topic_id in "${topic_ids[@]}"; do
  experiment_id="${run_id}-${topic_id}"
  shard_config="$config_root/${experiment_id}.yaml"
  work_root="$run_root/$topic_id"
  cache_root="$work_root/cache"
  mkdir -p "$work_root" "$cache_root"
  chmod 700 "$work_root" "$cache_root"
  validate_configured_topic "$venv_python" "$topic_id"
  "$venv_python" - "$source_config" "$shard_config" "$experiment_id" "$topic_id" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys

import yaml

from trec_rag.facet_pilot_config import load_facet_pilot_config, select_configured_topics

source, destination, experiment_id, topic_id = sys.argv[1:]
value = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
if not isinstance(value, dict) or not isinstance(value.get("experiment"), dict):
    raise SystemExit("source config has no experiment mapping")
value["experiment"]["id"] = experiment_id
execution = value.setdefault("execution", {})
if not isinstance(execution, dict):
    raise SystemExit("source config execution value is not a mapping")
execution["topic_workers"] = 1
target = Path(destination)
target.write_text(
    yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
loaded = load_facet_pilot_config(target)
selected = select_configured_topics(loaded, topic_ids=(topic_id,))
if len(selected) != 1 or selected[0].id != topic_id:
    raise SystemExit("selected topic is absent from the canonical source")
PY
done

# Authenticate storage once, then fail closed unless every selected immutable
# prefix is empty before model downloads, CUDA work, or hosted retrieval.
hf_cli buckets --help >/dev/null
hf_cli auth whoami --format json >/dev/null
bucket_info="$shared_root/bucket-info.json"
hf_cli buckets info "$BUCKET_ID" --format json >"$bucket_info"
"$venv_python" - "$bucket_info" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(value, dict) or value.get("private") is not True:
    raise SystemExit("the configured Hugging Face Bucket is not private")
PY

run_bucket_path="${BUCKET_REPO_PREFIX}/experiments/${run_id}"
run_remote_prefix="hf://buckets/${BUCKET_ID}/${run_bucket_path}"
remote_listing_before="$shared_root/remote-listing-before.json"
hf_cli buckets list "$run_remote_prefix" --recursive --format json >"$remote_listing_before"
for topic_id in "${topic_ids[@]}"; do
  topic_bucket_path="${run_bucket_path}/${topic_id}"
  "$venv_python" -m trec_rag.hf_bucket_listing \
    require-empty "$remote_listing_before" \
    --topic-prefix "$topic_bucket_path"
done

# Fetch both exact public snapshots before paid retrieval. The second lookup is
# offline and exercises the same local cache contract used by production.
"$venv_python" - "$MIXEDBREAD_REVISION" "$MINILM_REVISION" <<'PY'
from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from trec_rag.evidence_local import MINILM_MODEL, MINILM_REVISION
from trec_rag.mixedbread_passage_scorer import (
    MIXEDBREAD_MODEL,
    MIXEDBREAD_REVISION,
)

expected_mixedbread, expected_minilm = sys.argv[1:]
if MIXEDBREAD_REVISION != expected_mixedbread:
    raise SystemExit("Mixedbread revision drifted from the shard contract")
if MINILM_REVISION != expected_minilm:
    raise SystemExit("MiniLM revision drifted from the shard contract")
for label, model, revision in (
    ("reranker", MIXEDBREAD_MODEL, MIXEDBREAD_REVISION),
    ("similarity", MINILM_MODEL, MINILM_REVISION),
):
    snapshot_download(model, revision=revision)
    snapshot_download(model, revision=revision, local_files_only=True)
    print(f"model_snapshot_verified={label}@{revision}")
PY

"$venv_python" - <<'PY'
from __future__ import annotations

import sys
import torch

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("CUDA is unavailable", file=sys.stderr)
    raise SystemExit(2)
print(f"torch.cuda.device_name={torch.cuda.get_device_name(0)}")
PY

run_topic_job() (
  set -euo pipefail
  local topic_id=$1
  local experiment_id="${run_id}-${topic_id}"
  local shard_config="$config_root/${experiment_id}.yaml"
  local work_root="$run_root/$topic_id"
  local cache_root="$work_root/cache"
  local bundle_dir="$work_root/bundle"
  local topic_bucket_path="${run_bucket_path}/${topic_id}"
  local remote_prefix="hf://buckets/${BUCKET_ID}/${topic_bucket_path}"
  export TREC_RAG_CACHE_ROOT="$cache_root"

  "$venv_python" -m trec_rag.competition_retrieval \
    "$shard_config" \
    --topic "$topic_id"

  "$venv_python" -m trec_rag.competition_cache_bundle pack \
    --config "$shard_config" \
    --topic "$topic_id" \
    --destination "$bundle_dir"
  "$venv_python" -m trec_rag.competition_cache_bundle verify "$bundle_dir"

  local bundle_archive="$bundle_dir/bundle.tar.zst"
  local bundle_completion="$bundle_dir/bundle-complete.json"
  [[ -f $bundle_archive && ! -L $bundle_archive ]] || die "bundle archive is missing for topic $topic_id"
  [[ -f $bundle_completion && ! -L $bundle_completion ]] || die "bundle completion is missing for topic $topic_id"

  # The early check is not a reservation. Recheck immediately before the first
  # immutable upload so a concurrent writer cannot be silently ignored.
  local remote_listing_preupload="$work_root/remote-listing-preupload.json"
  hf_cli buckets list "$run_remote_prefix" --recursive --format json >"$remote_listing_preupload"
  "$venv_python" -m trec_rag.hf_bucket_listing \
    require-empty "$remote_listing_preupload" \
    --topic-prefix "$topic_bucket_path"

  local upload_number=0
  upload_one() {
    local source=$1
    local basename
    basename=$(basename "$source")
    upload_number=$((upload_number + 1))
    local upload_stage="$work_root/upload-${upload_number}"
    local roundtrip="$work_root/roundtrip-${upload_number}-${basename}"
    mkdir -m 700 "$upload_stage"
    cp -- "$source" "$upload_stage/$basename"
    hf_cli buckets sync "$upload_stage" "$remote_prefix" --ignore-existing
    hf_cli buckets cp "$remote_prefix/$basename" "$roundtrip"
    cmp -s -- "$source" "$roundtrip" \
      || die "remote $basename differs after upload for topic $topic_id"
  }

  # A consumer treats the prefix as complete only after the second publication.
  # No delete operation is supported or needed for this immutable workflow.
  upload_one "$bundle_archive"
  upload_one "$bundle_completion"

  local remote_listing_after="$work_root/remote-listing-after.json"
  hf_cli buckets list "$run_remote_prefix" --recursive --format json >"$remote_listing_after"
  "$venv_python" -m trec_rag.hf_bucket_listing \
    require-bundle "$remote_listing_after" \
    --topic-prefix "$topic_bucket_path"

  local downloaded_bundle="$work_root/downloaded-bundle"
  mkdir -m 700 "$downloaded_bundle"
  hf_cli buckets cp "$remote_prefix/bundle.tar.zst" "$downloaded_bundle/bundle.tar.zst"
  hf_cli buckets cp "$remote_prefix/bundle-complete.json" "$downloaded_bundle/bundle-complete.json"
  cmp -s -- "$bundle_archive" "$downloaded_bundle/bundle.tar.zst" \
    || die "downloaded archive digest differs for topic $topic_id"
  cmp -s -- "$bundle_completion" "$downloaded_bundle/bundle-complete.json" \
    || die "downloaded completion digest differs for topic $topic_id"
  "$venv_python" -m trec_rag.competition_cache_bundle verify "$downloaded_bundle"

  printf '%s\n' \
    "shard_status=complete" \
    "topic_id=$topic_id" \
    "experiment_id=$experiment_id" \
    "remote_prefix=$remote_prefix"
)

topic_pids=()
for topic_id in "${topic_ids[@]}"; do
  run_topic_job "$topic_id" &
  topic_pids+=("$!")
done

failed_topics=()
for topic_index in "${!topic_pids[@]}"; do
  if wait "${topic_pids[$topic_index]}"; then
    continue
  else
    topic_status=$?
  fi
  topic_id=${topic_ids[$topic_index]}
  printf 'ERROR: topic shard failed: %s (exit %s)\n' "$topic_id" "$topic_status" >&2
  failed_topics+=("$topic_id")
done

if ((${#failed_topics[@]})); then
  failed_topics_csv=$(IFS=,; printf '%s' "${failed_topics[*]}")
  die "one or more topic shards failed: $failed_topics_csv"
fi

printf '%s\n' \
  "shard_batch_status=complete" \
  "topic_ids=$topic_ids_csv" \
  "parallel_topic_processes=${#topic_ids[@]}"
