from __future__ import annotations

from pathlib import Path
import re
import shlex
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
        "hf://buckets/Npatta01/trec-rag-2026-artifacts/trec_rag_2026/artifacts/rag25-segmentation-baseline-20260807",
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
    assert "topic_workers=20" in result.stdout
    assert "source_verify_workers=8" in result.stdout
    assert "planning_calls_expected=0" in result.stdout
    assert "retrieval_network_calls_expected=0" in result.stdout
    assert "passage_model_batches_expected=0" in result.stdout
    assert "candidate_judge_calls_max=22" in result.stdout
    assert "topic_407_gate=post-full-structural" in result.stdout
    assert "warm_probe_topics=14,31" in result.stdout
    assert "warm_probe_seconds=900" in result.stdout


def test_live_wrapper_validates_topics_after_creating_the_locked_environment() -> None:
    wrapper = WRAPPER.read_text()

    assert 'validate_topic_order "$project_python"' in wrapper
    assert 'validate_topic_order "$venv_python"' in wrapper
    assert wrapper.index("uv sync --group cuda") < wrapper.index(
        'validate_topic_order "$venv_python"'
    )


def test_task_uses_one_fast_bounded_on_demand_gpu_and_only_named_secrets() -> None:
    value = yaml.safe_load(TASK.read_text())

    assert value["type"] == "task"
    assert value["image"].startswith("huggingface/trl@sha256:")
    assert value["resources"]["gpu"]["count"] == 1
    assert value["resources"]["gpu"]["memory"] == "80GB.."
    assert value["resources"]["gpu"]["name"] == ["H200", "H100"]
    assert value["resources"]["cpu"] == "24.."
    assert value["resources"]["memory"] == "192GB.."
    assert value["resources"]["disk"] == "100GB"
    assert value["backends"] == ["runpod", "vastai"]
    assert value["spot_policy"] == "on-demand"
    assert value["max_duration"] == "170m"
    assert value["commands"] == [
        "timeout --signal=TERM --kill-after=10m 160m "
        "bash code/tools/run_cached_segmentation_validation.sh ${{ run.args }}"
    ]
    assert value["max_price"] == 3.29
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


def test_expected_short_probe_timeout_is_success() -> None:
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
            f"set -euo pipefail\n{match.group(0)}\npropagate_probe_failure 124",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0


def test_source_verification_batch_waits_all_and_preserves_failure(
    tmp_path: Path,
) -> None:
    wrapper = WRAPPER.read_text()
    match = re.search(
        r"(?ms)^wait_source_bundle_batch\(\) \{\n.*?^\}\n",
        wrapper,
    )
    assert match is not None
    completed = tmp_path / "completed"

    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f"set -u\n{match.group(0)}\n"
                "(exit 7) & failed=$!\n"
                f"(sleep 0.05; touch {shlex.quote(str(completed))}) & completed=$!\n"
                'wait_source_bundle_batch "$failed" "$completed"\n'
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 7
    assert completed.is_file()


def test_launcher_preview_accepts_five_empty_approval_fields() -> None:
    launcher = LAUNCHER.read_text()
    match = re.search(
        r"(?ms)^approval_fields_empty\(\) \{\n.*?^\}\n",
        launcher,
    )
    assert match is not None

    empty = subprocess.run(
        [
            "bash",
            "-c",
            f"{match.group(0)}\napproval_fields_empty '' '' '' '' ''",
        ],
        check=False,
    )
    populated = subprocess.run(
        [
            "bash",
            "-c",
            f"{match.group(0)}\napproval_fields_empty '' runpod '' '' ''",
        ],
        check=False,
    )

    assert empty.returncode == 0
    assert populated.returncode == 1


def test_launcher_accepts_safe_cloud_instance_names_with_spaces() -> None:
    launcher = LAUNCHER.read_text()
    match = re.search(
        r"(?ms)^valid_instance_type\(\) \{\n.*?^\}\n",
        launcher,
    )
    assert match is not None

    valid = subprocess.run(
        [
            "bash",
            "-c",
            f"{match.group(0)}\nvalid_instance_type 'NVIDIA H100 80GB HBM3'",
        ],
        check=False,
    )
    leading_space = subprocess.run(
        ["bash", "-c", f"{match.group(0)}\nvalid_instance_type ' unsafe'"],
        check=False,
    )
    shell_syntax = subprocess.run(
        ["bash", "-c", f"{match.group(0)}\nvalid_instance_type '$(unsafe)'"],
        check=False,
    )

    assert valid.returncode == 0
    assert leading_space.returncode == 1
    assert shell_syntax.returncode == 1


def test_workflow_source_encodes_short_probe_full_run_and_marker_last() -> None:
    wrapper = WRAPPER.read_text()
    launcher = LAUNCHER.read_text()

    assert "competition_cache_bundle verify-rescore-source" in wrapper
    assert "competition_cache_bundle merge-rescore-source" in wrapper
    assert "SOURCE_VERIFY_WORKERS=8" in wrapper
    assert 'download_and_verify_source_bundle "$topic_id" &' in wrapper
    assert 'wait_source_bundle_batch "${source_verify_pids[@]}"' in wrapper
    authentication = wrapper.index(
        'wait_source_bundle_batch "${source_verify_pids[@]}"'
    )
    merge = wrapper.index("competition_cache_bundle merge")
    assert authentication < merge
    assert "--cached-upstream-rescore" in wrapper
    assert '"407"' in wrapper
    assert '"14" "31"' in wrapper
    assert "topic_workers: 4" not in wrapper  # generated structurally, not patched text
    assert "cached_segmentation_validation structural" in wrapper
    assert "cached_segmentation_validation semantic" in wrapper
    assert '--output-dir "$validation_root" --workers 8 "${topic_args[@]}"' in wrapper
    assert wrapper.count('--document-store-root "$cache_root/documents/v1"') == 1
    assert "cached-segmentation-concurrency-decision-v3" in wrapper
    assert "peak_memory_mib" in wrapper
    assert "projected_selected_worker_memory_mib" in wrapper
    assert "nvidia-smi --query-gpu=memory.used,memory.total" in wrapper
    assert "nvidia-smi --query-gpu=name,uuid,driver_version" in wrapper
    assert '"config_sha256": sha256(Path(config_path).read_bytes()).hexdigest()' in wrapper
    assert '"run_id": run_id' in wrapper
    assert '"gpu_uuid": gpu_uuid' in wrapper
    assert "SELECTED_WORKERS=20" in wrapper
    assert 'make_config "$final_config" "$run_id" "$SELECTED_WORKERS"' in wrapper
    assert "PROBE_DURATION_SECONDS=900" in wrapper
    assert "MINIMUM_PROBE_PEAK_DELTA_MIB=4096" in wrapper
    assert 'probe_cache_root="$work_root/probe-cache"' in wrapper
    assert 'make_config "$probe_config" "${run_id}-probe" "$PROBE_WORKERS"' in wrapper
    assert 'timeout --signal=TERM --kill-after=30s "${PROBE_DURATION_SECONDS}s"' in wrapper
    assert (
        '"$venv_python" -m trec_rag.competition_retrieval "$probe_config" \\\n'
        '  --topic 14 --topic 31 --cached-upstream-rescore'
    ) in wrapper
    assert (
        '"$venv_python" -m trec_rag.competition_retrieval "$final_config" \\\n'
        '  --cached-upstream-rescore'
    ) in wrapper
    assert '--topic 407 --cached-upstream-rescore' not in wrapper
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
    assert 'Decimal("3.29")' in launcher
    assert "approved_max_exposure=" in launcher
    assert 'Decimal("17") / Decimal("6")' in launcher
    assert 'remote set-url origin "$remote_url"' in launcher
    assert 'update-ref "refs/remotes/origin/$tracking_branch" "$tracking_head"' in launcher
    assert 'branch --set-upstream-to="origin/$tracking_branch"' in launcher
    assert "modal" not in wrapper.casefold()
    assert "modal" not in launcher.casefold()
