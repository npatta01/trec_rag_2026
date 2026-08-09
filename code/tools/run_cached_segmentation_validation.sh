#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUCKET_ID="Npatta01/trec_mlm_2026"
BUCKET_PREFIX="trec_rag_2026"
TOPICS=("14" "31" "37" "58" "72" "84" "144" "161" "200" "213" "219" "224" "225" "233" "273" "300" "407" "477" "499" "515" "707" "897")
MIXEDBREAD_REVISION="3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
MINILM_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SOURCE_VERIFY_WORKERS=8

preflight=false
run_id=""
source_run_id=""
baseline_uri=""
config_arg=""

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

propagate_probe_failure() {
  local status=$1
  if [[ $status == 0 ]]; then
    return 0
  fi
  printf 'ERROR: two-topic concurrency probe failed with status %s\n' \
    "$status" >&2
  return "$status"
}

while (($#)); do
  case "$1" in
    --preflight) preflight=true; shift ;;
    --run-id|--source-run-id|--baseline-uri|--config)
      (($# >= 2)) || die "$1 needs a value"
      case "$1" in
        --run-id) run_id=$2 ;;
        --source-run-id) source_run_id=$2 ;;
        --baseline-uri) baseline_uri=$2 ;;
        --config) config_arg=$2 ;;
      esac
      shift 2
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || die "--run-id is unsafe"
[[ $source_run_id =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] \
  || die "--source-run-id is unsafe"
expected_baseline="hf://buckets/${BUCKET_ID}/${BUCKET_PREFIX}/artifacts/rag25-segmentation-baseline-20260807"
[[ $baseline_uri == "$expected_baseline" ]] || die "--baseline-uri changed"
case "$config_arg" in /*|*..*) die "--config must be repository-relative" ;; esac

cd "$REPO_ROOT"
[[ $(git rev-parse --show-toplevel) == "$REPO_ROOT" ]] || die "unexpected Git root"
config_rel=$(git ls-files --full-name -- "$config_arg")
[[ -n $config_rel && $config_rel != *$'\n'* ]] || die "--config must be tracked"
source_config="$REPO_ROOT/$config_rel"
[[ -f $source_config && ! -L $source_config ]] || die "config is missing or unsafe"
for command_name in bash git python3 uv; do
  command -v "$command_name" >/dev/null || die "$command_name is required"
done
topics_csv=$(IFS=,; printf '%s' "${TOPICS[*]}")
validate_topic_order() {
  local interpreter=$1
  "$interpreter" - "$source_config" "$topics_csv" <<'PY'
from pathlib import Path
import sys
from trec_rag.facet_pilot_config import load_facet_pilot_config, select_configured_topics

source, expected = sys.argv[1:]
actual = tuple(topic.id for topic in select_configured_topics(load_facet_pilot_config(Path(source))))
if actual != tuple(expected.split(",")):
    raise SystemExit(f"configured topic order changed: {actual!r}")
PY
}

project_python="$REPO_ROOT/.venv/bin/python"
if $preflight; then
  [[ -x $project_python ]] || die "project Python is required for preflight"
  validate_topic_order "$project_python"
fi

source_prefix="hf://buckets/${BUCKET_ID}/${BUCKET_PREFIX}/experiments/${source_run_id}"
result_prefix="hf://buckets/${BUCKET_ID}/${BUCKET_PREFIX}/experiments/${run_id}"
diagnostic_prefix="${result_prefix}-diagnostic"
if $preflight; then
  printf '%s\n' \
    "preflight=ok" \
    "topic_ids=$topics_csv" \
    "topic_count=22" \
    "topic_workers=4" \
    "source_verify_workers=$SOURCE_VERIFY_WORKERS" \
    "canary_topic=407" \
    "warm_probe_topics=14,31" \
    "source_bundle_download_gib=1.30" \
    "source_prefix=$source_prefix" \
    "baseline_prefix=$baseline_uri" \
    "result_prefix=$result_prefix" \
    "diagnostic_prefix=$diagnostic_prefix" \
    "planning_calls_expected=0" \
    "retrieval_network_calls_expected=0" \
    "passage_model_batches_expected=0" \
    "candidate_judge_calls_max=22"
  exit 0
fi

for secret_name in HF_TOKEN INDEX_URL PYSERINI_API_TOKEN OPENROUTER_API_KEY; do
  [[ -n ${!secret_name:-} ]] || die "required dstack secret is missing: $secret_name"
done
[[ ${HF_CLI_MODE:-direct} == direct ]] || die "HF_CLI_MODE must be direct"

# Seal dstack's transported patch into a clean ephemeral commit so the runner's
# source identity and dirty-worktree checks remain meaningful.
git config user.name >/dev/null 2>&1 || git config user.name "dstack segmentation validation"
git config user.email >/dev/null 2>&1 || git config user.email "dstack-segmentation@invalid.local"
git add -A
git --no-pager diff --cached --check
if ! git diff --cached --quiet; then
  git commit --no-gpg-sign -m "dstack cached segmentation validation ${run_id}" >/dev/null
fi
git submodule update --init --recursive
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] \
  || die "transported checkout is not clean"

image_python=$(command -v python3)
uv sync --group cuda --locked --no-managed-python --no-python-downloads \
  --python "$image_python"
venv_python="$REPO_ROOT/.venv/bin/python"
venv_hf="$REPO_ROOT/.venv/bin/hf"
[[ -x $venv_python && -x $venv_hf ]] || die "locked environment setup failed"
hf_cli() { "$venv_hf" "$@"; }
validate_topic_order "$venv_python"

work_root="/tmp/trec-rag-segmentation/${run_id}"
bundle_root="$work_root/source-bundles"
cache_root="$work_root/cache"
source_outputs="$work_root/source-outputs"
baseline_bundle="$work_root/baseline-bundle"
baseline_restore="$work_root/baseline"
baseline_output="$work_root/baseline-output"
validation_root="$work_root/validation"
result_bundle="$work_root/result-bundle"
roundtrip_bundle="$work_root/roundtrip-bundle"
diagnostic_bundle="$work_root/diagnostic-bundle"
diagnostic_roundtrip_bundle="$work_root/diagnostic-roundtrip-bundle"
config_root="$REPO_ROOT/configs/local"
final_output="$REPO_ROOT/outputs/$run_id"
mkdir -p "$bundle_root" "$cache_root" "$source_outputs" "$baseline_bundle" \
  "$baseline_output" "$validation_root" "$result_bundle" "$roundtrip_bundle" \
  "$diagnostic_bundle" "$diagnostic_roundtrip_bundle" "$config_root"
chmod 700 "$work_root" "$cache_root" "$source_outputs" "$validation_root"
export TREC_RAG_CACHE_ROOT="$cache_root"
export HF_HOME="$work_root/huggingface"

diagnostic_enabled=false
diagnostic_stage="authenticated-input-download"
diagnostic_upload_number=0
upload_diagnostic() {
  local source=$1
  diagnostic_upload_number=$((diagnostic_upload_number + 1))
  local upload_stage="$work_root/diagnostic-upload-$diagnostic_upload_number"
  mkdir -m 700 "$upload_stage"
  cp -- "$source" "$upload_stage/$(basename "$source")"
  hf_cli buckets sync "$upload_stage" "$diagnostic_prefix" --ignore-existing
}
preserve_failure() {
  local original_status=$?
  trap - EXIT
  if [[ $original_status == 0 || $diagnostic_enabled != true ]]; then
    exit "$original_status"
  fi
  set +e
  printf 'Preserving non-promotable diagnostic evidence for stage %s\n' \
    "$diagnostic_stage" >&2
  local revision
  revision=$(git rev-parse HEAD)
  local diagnostic_args=()
  local topic_id
  for topic_id in "${TOPICS[@]}"; do diagnostic_args+=(--topic "$topic_id"); done
  "$venv_python" -m trec_rag.cached_segmentation_result_bundle pack-diagnostic \
    --run-root "$final_output" --validation-root "$validation_root" \
    --destination "$diagnostic_bundle" --run-id "$run_id" \
    --revision "$revision" --stage "$diagnostic_stage" \
    --exit-status "$original_status" "${diagnostic_args[@]}"
  local diagnostic_status=$?
  if [[ $diagnostic_status == 0 ]]; then
    "$venv_python" -m trec_rag.cached_segmentation_result_bundle \
      verify-diagnostic "$diagnostic_bundle"
    diagnostic_status=$?
  fi
  local diagnostic_archive="$diagnostic_bundle/bundle.tar.zst"
  local diagnostic_completion="$diagnostic_bundle/bundle-complete.json"
  if [[ $diagnostic_status == 0 ]]; then
    upload_diagnostic "$diagnostic_archive"
    diagnostic_status=$?
  fi
  if [[ $diagnostic_status == 0 ]]; then
    upload_diagnostic "$diagnostic_completion"
    diagnostic_status=$?
  fi
  if [[ $diagnostic_status == 0 ]]; then
    hf_cli buckets cp "$diagnostic_prefix/bundle.tar.zst" \
      "$diagnostic_roundtrip_bundle/bundle.tar.zst"
    diagnostic_status=$?
  fi
  if [[ $diagnostic_status == 0 ]]; then
    hf_cli buckets cp "$diagnostic_prefix/bundle-complete.json" \
      "$diagnostic_roundtrip_bundle/bundle-complete.json"
    diagnostic_status=$?
  fi
  if [[ $diagnostic_status == 0 ]]; then
    cmp -s "$diagnostic_archive" \
      "$diagnostic_roundtrip_bundle/bundle.tar.zst" \
      && cmp -s "$diagnostic_completion" \
        "$diagnostic_roundtrip_bundle/bundle-complete.json" \
      && "$venv_python" -m trec_rag.cached_segmentation_result_bundle \
        verify-diagnostic "$diagnostic_roundtrip_bundle"
    diagnostic_status=$?
  fi
  if [[ $diagnostic_status == 0 ]]; then
    printf '%s\n' "diagnostic_status=complete" \
      "diagnostic_prefix=$diagnostic_prefix" >&2
  else
    printf 'ERROR: diagnostic preservation failed with status %s\n' \
      "$diagnostic_status" >&2
  fi
  exit "$original_status"
}

hf_cli buckets --help >/dev/null
hf_cli auth whoami --format json >/dev/null
hf_cli buckets info "$BUCKET_ID" --format json >"$work_root/bucket-info.json"
"$venv_python" - "$work_root/bucket-info.json" <<'PY'
import json, sys
from pathlib import Path
if json.loads(Path(sys.argv[1]).read_text()).get("private") is not True:
    raise SystemExit("configured Hugging Face Bucket is not private")
PY

hf_cli buckets list "$source_prefix" --recursive --format json >"$work_root/source-listing.json"
for topic_id in "${TOPICS[@]}"; do
  "$venv_python" -m trec_rag.hf_bucket_listing require-bundle \
    "$work_root/source-listing.json" \
    --topic-prefix "${BUCKET_PREFIX}/experiments/${source_run_id}/${topic_id}"
done
hf_cli buckets list "$baseline_uri" --recursive --format json >"$work_root/baseline-listing.json"
"$venv_python" -m trec_rag.hf_bucket_listing require-bundle \
  "$work_root/baseline-listing.json" \
  --topic-prefix "${BUCKET_PREFIX}/artifacts/rag25-segmentation-baseline-20260807"
hf_cli buckets list "$result_prefix" --recursive --format json >"$work_root/result-before.json"
"$venv_python" -m trec_rag.hf_bucket_listing require-empty \
  "$work_root/result-before.json" \
  --topic-prefix "${BUCKET_PREFIX}/experiments/${run_id}"
hf_cli buckets list "$diagnostic_prefix" --recursive --format json \
  >"$work_root/diagnostic-before.json"
"$venv_python" -m trec_rag.hf_bucket_listing require-empty \
  "$work_root/diagnostic-before.json" \
  --topic-prefix "${BUCKET_PREFIX}/experiments/${run_id}-diagnostic"
diagnostic_enabled=true
trap preserve_failure EXIT

bundle_dirs=()
source_verify_pids=()
download_and_verify_source_bundle() {
  local topic_id=$1
  local topic_bundle="$bundle_root/$topic_id"
  mkdir -m 700 "$topic_bundle"
  hf_cli buckets cp \
    "$source_prefix/$topic_id/bundle.tar.zst" \
    "$topic_bundle/bundle.tar.zst"
  hf_cli buckets cp \
    "$source_prefix/$topic_id/bundle-complete.json" \
    "$topic_bundle/bundle-complete.json"
  "$venv_python" -m trec_rag.competition_cache_bundle verify "$topic_bundle"
}
wait_source_bundle_batch() {
  local first_status=0
  local status
  local pid
  for pid in "$@"; do
    if wait "$pid"; then
      continue
    else
      status=$?
    fi
    if ((first_status == 0)); then
      first_status=$status
    fi
  done
  if ((first_status != 0)); then
    return "$first_status"
  fi
}
for topic_id in "${TOPICS[@]}"; do
  topic_bundle="$bundle_root/$topic_id"
  bundle_dirs+=("$topic_bundle")
  download_and_verify_source_bundle "$topic_id" &
  source_verify_pids+=("$!")
  if ((${#source_verify_pids[@]} == SOURCE_VERIFY_WORKERS)); then
    wait_source_bundle_batch "${source_verify_pids[@]}"
    source_verify_pids=()
  fi
done
if ((${#source_verify_pids[@]})); then
  wait_source_bundle_batch "${source_verify_pids[@]}"
fi
# All 22 are authenticated before a single destination mutation.
"$venv_python" -m trec_rag.competition_cache_bundle merge \
  --cache-root "$cache_root" --outputs-root "$source_outputs" \
  "${bundle_dirs[@]}"

hf_cli buckets cp "$baseline_uri/bundle.tar.zst" "$baseline_bundle/bundle.tar.zst"
hf_cli buckets cp "$baseline_uri/bundle-complete.json" "$baseline_bundle/bundle-complete.json"
"$venv_python" -m trec_rag.cached_segmentation_result_bundle verify-baseline "$baseline_bundle"
"$venv_python" -m trec_rag.cached_segmentation_result_bundle restore-baseline \
  "$baseline_bundle" --destination "$baseline_restore"

# Build one stable per-topic view of the old authenticated checkpoints restored
# from the 22 portable source shards.
"$venv_python" - "$source_outputs" "$baseline_output" "$topics_csv" <<'PY'
from pathlib import Path
import os, sys
source, destination, topics = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
for topic in topics:
    matches = [path for path in source.glob(f"*/{topic}") if (path / "scoring/complete.json").is_file()]
    if len(matches) != 1:
        raise SystemExit(f"expected one restored source checkpoint for topic {topic}, found {matches}")
    os.symlink(matches[0], destination / topic, target_is_directory=True)
PY

"$venv_python" - "$MIXEDBREAD_REVISION" "$MINILM_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download
from trec_rag.evidence_local import MINILM_MODEL, MINILM_REVISION
from trec_rag.mixedbread_passage_scorer import MIXEDBREAD_MODEL, MIXEDBREAD_REVISION
expected_reranker, expected_similarity = sys.argv[1:]
if (MIXEDBREAD_REVISION, MINILM_REVISION) != (expected_reranker, expected_similarity):
    raise SystemExit("pinned model revision changed")
for model, revision in ((MIXEDBREAD_MODEL, MIXEDBREAD_REVISION), (MINILM_MODEL, MINILM_REVISION)):
    snapshot_download(model, revision=revision)
    snapshot_download(model, revision=revision, local_files_only=True)
PY
"$venv_python" - <<'PY'
import sys, torch
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
nvidia-smi

make_config() {
  local destination=$1 experiment_id=$2 workers=$3
  "$venv_python" - "$source_config" "$destination" "$experiment_id" "$workers" <<'PY'
from pathlib import Path
import sys, yaml
from trec_rag.facet_pilot_config import load_facet_pilot_config, select_configured_topics
source, destination, experiment_id, workers = sys.argv[1:]
value = yaml.safe_load(Path(source).read_text())
value["experiment"]["id"] = experiment_id
value["execution"]["topic_workers"] = int(workers)
target = Path(destination)
target.write_text(yaml.safe_dump(value, sort_keys=False))
loaded = load_facet_pilot_config(target)
if len(select_configured_topics(loaded)) != 22:
    raise SystemExit("generated config topic set changed")
PY
}

final_config="$config_root/${run_id}.yaml"
make_config "$final_config" "$run_id" 4

diagnostic_stage="canary-rescore"
"$venv_python" -m trec_rag.competition_retrieval "$final_config" \
  --topic 407 --cached-upstream-rescore
baseline_handoff="$baseline_restore/generation_handoff_manifest.json"
baseline_coverage="$baseline_restore/retrieval_nugget_coverage_v2"
"$venv_python" - "$baseline_handoff" "$work_root/baseline-407.json" <<'PY'
from pathlib import Path
import sys
from trec_rag.generation_handoff import GenerationHandoff, load_generation_handoff, write_generation_handoff
source, destination = map(Path, sys.argv[1:])
loaded = load_generation_handoff(source)
topics = tuple(topic for topic in loaded.topics if topic.topic_id == "407")
if len(topics) != 1:
    raise SystemExit("baseline topic 407 changed")
write_generation_handoff(destination, GenerationHandoff(loaded.producer, topics))
PY
diagnostic_stage="canary-structural-validation"
"$venv_python" -m trec_rag.cached_segmentation_validation structural \
  --baseline-output-root "$baseline_output" \
  --candidate-output-root "$final_output" \
  --document-store-root "$cache_root/documents/v1" \
  --baseline-handoff "$work_root/baseline-407.json" \
  --candidate-handoff "$final_output/generation_handoff_manifest.json" \
  --output-dir "$validation_root/canary" --topic 407

# The global cache-operation marker covers the selected invocation, while the
# topic receipts are immutable and resumable. Remove only that expandable root
# marker before extending this same run namespace with the probe and full set.
"$venv_python" - "$final_output/cache-operation-manifest.json" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    raise SystemExit("canary cache-operation marker is missing or unsafe")
path.unlink()
PY

diagnostic_stage="concurrency-probe"
probe_memory="$validation_root/warm-probe-memory.csv"
gpu_identity="$validation_root/gpu-identity.csv"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader \
  >"$gpu_identity"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
  >"$probe_memory"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
  --loop=1 >>"$probe_memory" &
monitor_pid=$!
probe_started=$(date +%s)
set +e
"$venv_python" -m trec_rag.competition_retrieval "$final_config" \
  --topic 407 --topic 14 --topic 31 --cached-upstream-rescore
probe_status=$?
set -e
probe_finished=$(date +%s)
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
propagate_probe_failure "$probe_status"

concurrency_decision="$validation_root/concurrency-decision.json"
"$venv_python" - "$probe_memory" "$concurrency_decision" \
  "$probe_started" "$probe_finished" "$run_id" "$final_config" \
  "$gpu_identity" <<'PY'
from pathlib import Path
from hashlib import sha256
import csv, json, sys
source, destination = map(Path, sys.argv[1:3])
started, finished = map(int, sys.argv[3:5])
run_id, config_path, gpu_identity_path = sys.argv[5:]
samples = []
for line in source.read_text().splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 2:
        raise SystemExit("GPU probe row changed")
    samples.append(tuple(map(int, fields)))
if not samples or len({total for _, total in samples}) != 1:
    raise SystemExit("GPU probe samples are incomplete")
idle = samples[0][0]
peak = max(used for used, _ in samples)
total = samples[0][1]
projected = idle + 2 * max(0, peak - idle)
safe_limit = int(total * 0.90)
if projected > safe_limit:
    raise SystemExit(
        f"four-worker projection {projected} MiB exceeds 90% limit {safe_limit} MiB"
    )
with Path(gpu_identity_path).open(newline="") as source_file:
    identity_rows = [tuple(field.strip() for field in row) for row in csv.reader(source_file)]
if len(identity_rows) != 1 or len(identity_rows[0]) != 3:
    raise SystemExit("expected exactly one GPU identity")
gpu_name, gpu_uuid, driver_version = identity_rows[0]
if not any(model in gpu_name for model in ("H100", "H200")) or not gpu_uuid.startswith("GPU-"):
    raise SystemExit("GPU identity differs from the approved task class")
value = {
    "schema_version": "cached-segmentation-concurrency-decision-v1",
    "run_id": run_id,
    "config_sha256": sha256(Path(config_path).read_bytes()).hexdigest(),
    "gpu_name": gpu_name,
    "gpu_uuid": gpu_uuid,
    "driver_version": driver_version,
    "probe_topics": ["14", "31"],
    "probe_workers": 2,
    "selected_workers": 4,
    "elapsed_seconds": finished - started,
    "idle_memory_mib": idle,
    "peak_memory_mib": peak,
    "total_memory_mib": total,
    "projected_four_worker_memory_mib": projected,
    "safe_limit_memory_mib": safe_limit,
}
destination.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
PY

"$venv_python" - "$final_output/cache-operation-manifest.json" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    raise SystemExit("probe cache-operation marker is missing or unsafe")
path.unlink()
PY

diagnostic_stage="full-rescore"
"$venv_python" -m trec_rag.competition_retrieval "$final_config" \
  --cached-upstream-rescore

topic_args=()
for topic_id in "${TOPICS[@]}"; do topic_args+=(--topic "$topic_id"); done
diagnostic_stage="full-structural-validation"
"$venv_python" -m trec_rag.cached_segmentation_validation structural \
  --baseline-output-root "$baseline_output" \
  --candidate-output-root "$final_output" \
  --document-store-root "$cache_root/documents/v1" \
  --baseline-handoff "$baseline_handoff" \
  --candidate-handoff "$final_output/generation_handoff_manifest.json" \
  --output-dir "$validation_root" "${topic_args[@]}"
diagnostic_stage="semantic-validation"
"$venv_python" -m trec_rag.cached_segmentation_validation semantic \
  --baseline-handoff "$baseline_handoff" \
  --baseline-coverage-root "$baseline_coverage" \
  --candidate-handoff "$final_output/generation_handoff_manifest.json" \
  --candidate-coverage-root "$validation_root/retrieval_nugget_coverage_v2" \
  --output-dir "$validation_root" "${topic_args[@]}"

diagnostic_stage="result-pack"
"$venv_python" -m trec_rag.cached_segmentation_result_bundle pack \
  --run-root "$final_output" --validation-root "$validation_root" \
  --destination "$result_bundle"
"$venv_python" -m trec_rag.cached_segmentation_result_bundle verify "$result_bundle"
nvidia-smi

result_archive="$result_bundle/bundle.tar.zst"
result_completion="$result_bundle/bundle-complete.json"
upload_number=0
upload_one() {
  local source=$1
  upload_number=$((upload_number + 1))
  local stage="$work_root/upload-$upload_number"
  mkdir -m 700 "$stage"
  cp -- "$source" "$stage/$(basename "$source")"
  hf_cli buckets sync "$stage" "$result_prefix" --ignore-existing
}
diagnostic_stage="result-upload"
hf_cli buckets list "$result_prefix" --recursive --format json >"$work_root/result-preupload.json"
"$venv_python" -m trec_rag.hf_bucket_listing require-empty \
  "$work_root/result-preupload.json" \
  --topic-prefix "${BUCKET_PREFIX}/experiments/${run_id}"
upload_one "$result_archive"
upload_one "$result_completion"

diagnostic_stage="result-roundtrip"
hf_cli buckets cp "$result_prefix/bundle.tar.zst" "$roundtrip_bundle/bundle.tar.zst"
hf_cli buckets cp "$result_prefix/bundle-complete.json" "$roundtrip_bundle/bundle-complete.json"
cmp -s "$result_archive" "$roundtrip_bundle/bundle.tar.zst" \
  || die "round-trip result archive changed"
cmp -s "$result_completion" "$roundtrip_bundle/bundle-complete.json" \
  || die "round-trip result marker changed"
"$venv_python" -m trec_rag.cached_segmentation_result_bundle verify "$roundtrip_bundle"
printf '%s\n' "validation_status=complete" "result_prefix=$result_prefix"
