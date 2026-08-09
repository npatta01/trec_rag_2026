#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUCKET_ID="Npatta01/trec_mlm_2026"
BUCKET_PREFIX="trec_rag_2026"
TOPICS=("14" "31" "37" "58" "72" "84" "144" "161" "200" "213" "219" "224" "225" "233" "273" "300" "407" "477" "499" "515" "707" "897")
MIXEDBREAD_REVISION="3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
MINILM_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

preflight=false
run_id=""
source_run_id=""
baseline_uri=""
config_arg=""

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
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
project_python="$REPO_ROOT/.venv/bin/python"
[[ -x $project_python ]] || die "project Python is required"

topics_csv=$(IFS=,; printf '%s' "${TOPICS[*]}")
"$project_python" - "$source_config" "$topics_csv" <<'PY'
from pathlib import Path
import sys
from trec_rag.facet_pilot_config import load_facet_pilot_config, select_configured_topics

source, expected = sys.argv[1:]
actual = tuple(topic.id for topic in select_configured_topics(load_facet_pilot_config(Path(source))))
if actual != tuple(expected.split(",")):
    raise SystemExit(f"configured topic order changed: {actual!r}")
PY

source_prefix="hf://buckets/${BUCKET_ID}/${BUCKET_PREFIX}/experiments/${source_run_id}"
result_prefix="hf://buckets/${BUCKET_ID}/${BUCKET_PREFIX}/experiments/${run_id}"
if $preflight; then
  printf '%s\n' \
    "preflight=ok" \
    "topic_ids=$topics_csv" \
    "topic_count=22" \
    "topic_workers=4" \
    "canary_topic=407" \
    "warm_probe_topics=14,31" \
    "source_bundle_download_gib=1.30" \
    "source_prefix=$source_prefix" \
    "baseline_prefix=$baseline_uri" \
    "result_prefix=$result_prefix" \
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
config_root="$REPO_ROOT/configs/local"
mkdir -p "$bundle_root" "$cache_root" "$source_outputs" "$baseline_bundle" \
  "$baseline_output" "$validation_root" "$result_bundle" "$roundtrip_bundle" \
  "$config_root"
chmod 700 "$work_root" "$cache_root" "$source_outputs" "$validation_root"
export TREC_RAG_CACHE_ROOT="$cache_root"
export HF_HOME="$work_root/huggingface"

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

bundle_dirs=()
for topic_id in "${TOPICS[@]}"; do
  topic_bundle="$bundle_root/$topic_id"
  mkdir -m 700 "$topic_bundle"
  hf_cli buckets cp "$source_prefix/$topic_id/bundle.tar.zst" "$topic_bundle/bundle.tar.zst"
  hf_cli buckets cp "$source_prefix/$topic_id/bundle-complete.json" "$topic_bundle/bundle-complete.json"
  "$venv_python" -m trec_rag.competition_cache_bundle verify "$topic_bundle"
  bundle_dirs+=("$topic_bundle")
done
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

canary_config="$config_root/${run_id}-canary.yaml"
warm_config="$config_root/${run_id}-warm.yaml"
final_config="$config_root/${run_id}.yaml"
make_config "$canary_config" "${run_id}-canary" 1
make_config "$warm_config" "${run_id}-warm" 2
make_config "$final_config" "$run_id" 4

"$venv_python" -m trec_rag.competition_retrieval "$canary_config" \
  --topic 407 --cached-upstream-rescore
canary_output="$REPO_ROOT/outputs/${run_id}-canary"
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
"$venv_python" -m trec_rag.cached_segmentation_validation structural \
  --baseline-output-root "$baseline_output" \
  --candidate-output-root "$canary_output" \
  --document-store-root "$cache_root/documents" \
  --baseline-handoff "$work_root/baseline-407.json" \
  --candidate-handoff "$canary_output/generation_handoff_manifest.json" \
  --output-dir "$validation_root/canary" --topic 407

"$venv_python" -m trec_rag.competition_retrieval "$warm_config" \
  --topic 14 --topic 31 --cached-upstream-rescore
"$venv_python" -m trec_rag.competition_retrieval "$final_config" \
  --cached-upstream-rescore
final_output="$REPO_ROOT/outputs/$run_id"

topic_args=()
for topic_id in "${TOPICS[@]}"; do topic_args+=(--topic "$topic_id"); done
"$venv_python" -m trec_rag.cached_segmentation_validation structural \
  --baseline-output-root "$baseline_output" \
  --candidate-output-root "$final_output" \
  --document-store-root "$cache_root/documents" \
  --baseline-handoff "$baseline_handoff" \
  --candidate-handoff "$final_output/generation_handoff_manifest.json" \
  --output-dir "$validation_root" "${topic_args[@]}"
"$venv_python" -m trec_rag.cached_segmentation_validation semantic \
  --baseline-handoff "$baseline_handoff" \
  --baseline-coverage-root "$baseline_coverage" \
  --candidate-handoff "$final_output/generation_handoff_manifest.json" \
  --candidate-coverage-root "$validation_root/retrieval_nugget_coverage_v2" \
  --output-dir "$validation_root" "${topic_args[@]}"

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
hf_cli buckets list "$result_prefix" --recursive --format json >"$work_root/result-preupload.json"
"$venv_python" -m trec_rag.hf_bucket_listing require-empty \
  "$work_root/result-preupload.json" \
  --topic-prefix "${BUCKET_PREFIX}/experiments/${run_id}"
upload_one "$result_archive"
upload_one "$result_completion"

hf_cli buckets cp "$result_prefix/bundle.tar.zst" "$roundtrip_bundle/bundle.tar.zst"
hf_cli buckets cp "$result_prefix/bundle-complete.json" "$roundtrip_bundle/bundle-complete.json"
cmp -s "$result_archive" "$roundtrip_bundle/bundle.tar.zst" \
  || die "round-trip result archive changed"
cmp -s "$result_completion" "$roundtrip_bundle/bundle-complete.json" \
  || die "round-trip result marker changed"
"$venv_python" -m trec_rag.cached_segmentation_result_bundle verify "$roundtrip_bundle"
printf '%s\n' "validation_status=complete" "result_prefix=$result_prefix"
