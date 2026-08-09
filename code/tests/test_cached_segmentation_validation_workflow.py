from __future__ import annotations

from pathlib import Path
import re
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "code/tools/run_cached_segmentation_validation.sh"
LAUNCHER = REPO_ROOT / "code/tools/apply_cached_segmentation_validation.sh"
TASK = REPO_ROOT / ".dstack/rag25-cached-segmentation-validation.yaml"
TOPICS = (
    "14",
    "31",
    "37",
    "58",
    "72",
    "84",
    "144",
    "161",
    "200",
    "213",
    "219",
    "224",
    "225",
    "233",
    "273",
    "300",
    "407",
    "477",
    "499",
    "515",
    "707",
    "897",
)


def _args() -> list[str]:
    return [
        "--run-id",
        "nonagentic-rag25-segmentation-fixed-20260809",
        "--source-run-id",
        "nonagentic-rag25-dev-20260806",
        "--baseline-uri",
        "hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/artifacts/rag25-segmentation-baseline-20260807",
        "--config",
        "configs/rag25_competition_retrieval_v1.yaml",
    ]


def test_wrapper_preflight_freezes_all22_and_zero_upstream_expectations() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "--preflight", *_args()],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"topic_ids={','.join(TOPICS)}" in result.stdout
    assert "topic_count=22" in result.stdout
    assert "topic_workers=4" in result.stdout
    assert "planning_calls_expected=0" in result.stdout
    assert "retrieval_network_calls_expected=0" in result.stdout
    assert "passage_model_batches_expected=0" in result.stdout
    assert "candidate_judge_calls_max=22" in result.stdout
    assert "canary_topic=407" in result.stdout
    assert "warm_probe_topics=14,31" in result.stdout


def test_task_uses_one_fast_bounded_on_demand_gpu_and_only_named_secrets() -> None:
    value = yaml.safe_load(TASK.read_text())

    assert value["type"] == "task"
    assert value["image"].startswith("huggingface/trl@sha256:")
    assert value["resources"]["gpu"]["count"] == 1
    assert value["resources"]["gpu"]["memory"] == "80GB.."
    assert value["resources"]["gpu"]["name"] == ["H200", "H100"]
    assert value["resources"]["memory"] == "64GB.."
    assert value["resources"]["disk"] == "100GB"
    assert value["backends"] == ["runpod", "vastai"]
    assert value["spot_policy"] == "on-demand"
    assert value["max_duration"] == "5h"
    assert value["max_price"] == 3.0
    assert value["retry"] == {"on_events": ["no-capacity"], "duration": "30m"}
    assert value["env"] == [
        "HF_TOKEN=${{ secrets.hf_token }}",
        "INDEX_URL=${{ secrets.INDEX_URL }}",
        "PYSERINI_API_TOKEN=${{ secrets.PYSERINI_API_TOKEN }}",
        "OPENROUTER_API_KEY=${{ secrets.OPENROUTER_API_KEY }}",
        "HF_CLI_MODE=direct",
    ]
    assert value["repos"][0]["local_path"].startswith("/dev/null/")


def test_probe_failure_preserves_the_underlying_exit_status() -> None:
    wrapper = WRAPPER.read_text()
    match = re.search(
        r"(?ms)^propagate_probe_failure\(\) \{\n.*?^\}\n",
        wrapper,
    )
    assert match is not None

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"set -euo pipefail\n{match.group(0)}\npropagate_probe_failure 137",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 137
    assert "probe failed with status 137" in result.stderr


def test_workflow_source_encodes_verify_merge_canary_full_and_marker_last() -> None:
    wrapper = WRAPPER.read_text()
    launcher = LAUNCHER.read_text()

    assert "competition_cache_bundle verify" in wrapper
    assert "competition_cache_bundle merge" in wrapper
    assert "--cached-upstream-rescore" in wrapper
    assert '"407"' in wrapper
    assert '"14" "31"' in wrapper
    assert "topic_workers: 4" not in wrapper  # generated structurally, not patched text
    assert "cached_segmentation_validation structural" in wrapper
    assert "cached_segmentation_validation semantic" in wrapper
    assert wrapper.count('--document-store-root "$cache_root/documents/v1"') == 2
    assert "cached-segmentation-concurrency-decision-v1" in wrapper
    assert "peak_memory_mib" in wrapper
    assert "projected_four_worker_memory_mib" in wrapper
    assert "nvidia-smi --query-gpu=memory.used,memory.total" in wrapper
    assert "nvidia-smi --query-gpu=name,uuid,driver_version" in wrapper
    assert '"config_sha256": sha256(Path(config_path).read_bytes()).hexdigest()' in wrapper
    assert '"run_id": run_id' in wrapper
    assert '"gpu_uuid": gpu_uuid' in wrapper
    assert 'make_config "$final_config" "$run_id" 4' in wrapper
    assert (
        '"$venv_python" -m trec_rag.competition_retrieval "$final_config" \\\n'
        '  --topic 407 --topic 14 --topic 31 --cached-upstream-rescore'
    ) in wrapper
    assert 'outputs/${run_id}-canary' not in wrapper
    assert 'outputs/${run_id}-warm' not in wrapper
    assert "cached_segmentation_result_bundle pack" in wrapper
    assert "cached_segmentation_result_bundle pack-diagnostic" in wrapper
    assert "verify-diagnostic" in wrapper
    assert 'diagnostic_prefix="${result_prefix}-diagnostic"' in wrapper
    assert "preserve_failure" in wrapper
    assert "trap preserve_failure EXIT" in wrapper
    assert wrapper.index('upload_diagnostic "$diagnostic_archive"') < wrapper.index(
        'upload_diagnostic "$diagnostic_completion"'
    )
    assert wrapper.index('upload_one "$result_archive"') < wrapper.index(
        'upload_one "$result_completion"'
    )
    assert '"$dstack_bin" apply -f' in launcher
    assert "printf 'n\\n'" in launcher
    assert '"$dstack_bin" apply' in launcher
    assert "-y -d" in launcher
    assert "--approved-backend" in launcher
    assert "--approved-region" in launcher
    assert "--approved-instance-type" in launcher
    assert "--approved-gpu" in launcher
    assert "--approved-hourly-price" in launcher
    assert '"--backend" "$approved_backend"' in launcher
    assert '"--region" "$approved_region"' in launcher
    assert '"--instance-type" "$approved_instance_type"' in launcher
    assert '"--gpu" "${approved_gpu}:1"' in launcher
    assert '"--max-price" "$approved_hourly_price"' in launcher
    assert "modal" not in wrapper.casefold()
    assert "modal" not in launcher.casefold()
