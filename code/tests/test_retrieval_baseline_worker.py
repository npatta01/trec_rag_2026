from __future__ import annotations

from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = REPO_ROOT / "code/tools/run_retrieval_baseline_worker.sh"
LAUNCHER = REPO_ROOT / "code/tools/apply_retrieval_baseline_worker.sh"
TEMPLATE = REPO_ROOT / ".dstack/rag26-retrieval-baseline-worker.yaml"


def test_worker_credential_free_preflight_accepts_two_topics() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(WORKER),
            "--preflight",
            "--task-name",
            "smoke-0-37",
            "--input-prefix",
            "hf://buckets/private/trec_rag_2026/artifacts/baseline-smoke-v1",
            "--input-manifest-sha256",
            "a" * 64,
            "--output-prefix",
            "hf://buckets/private/trec_rag_2026/experiments/baseline-smoke-v1",
            "--topic",
            "rag2026-0",
            "--topic",
            "rag2026-37",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "preflight=ok" in completed.stdout
    assert "topic_ids=rag2026-0,rag2026-37" in completed.stdout
    assert f"input_manifest_sha256={'a' * 64}" in completed.stdout
    assert "sequential_topic_processes=1" in completed.stdout


def test_worker_preflight_rejects_duplicate_topics() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(WORKER),
            "--preflight",
            "--task-name",
            "duplicate",
            "--input-prefix",
            "hf://buckets/private/trec_rag_2026/artifacts/baseline-v1",
            "--input-manifest-sha256",
            "a" * 64,
            "--output-prefix",
            "hf://buckets/private/trec_rag_2026/experiments/baseline-v1",
            "--topic",
            "rag2026-0",
            "--topic",
            "rag2026-0",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "unique" in completed.stderr


def test_dstack_template_is_pinned_private_and_gpu_bounded() -> None:
    value = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))

    assert value["type"] == "task"
    assert "@sha256:" in value["image"]
    assert value["repos"][0]["local_path"].startswith("/dev/null/")
    assert value["env"] == [
        "HF_TOKEN=${{ secrets.hf_token }}",
        "HF_CLI_MODE=direct",
    ]
    assert value["resources"]["gpu"]["count"] == 1
    assert value["resources"]["gpu"]["name"] == ["B200"]
    assert value["resources"]["gpu"]["memory"] == "48GB.."
    assert value["spot_policy"] == "on-demand"
    assert value["idle_duration"] == "0s"
    assert value["max_price"] == 7.0


def test_launcher_defaults_to_declined_preview_and_uses_clean_snapshot() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "mode=preview" in source
    assert "printf 'n\\n'" in source
    assert 'git clone --quiet --no-hardlinks "$REPO_ROOT" "$snapshot"' in source
    assert '[[ -z $source_status ]]' in source
    assert '"$dstack_bin" apply' in source
    assert ' -y -d -- ' in source
    assert "--input-manifest-sha256" in source


def test_worker_checks_privacy_and_output_before_model_scoring() -> None:
    source = WORKER.read_text(encoding="utf-8")

    privacy_offset = source.index('buckets info "$bucket"')
    empty_offset = source.index('require_empty_prefix "$output_prefix"')
    model_offset = source.index("snapshot_download(model, revision=revision)")
    scoring_offset = source.index("retrieval_baseline_runs score-topic")
    assert privacy_offset < model_offset
    assert empty_offset < model_offset
    assert model_offset < scoring_offset
    assert "--no-managed-python" in source
    assert "--no-python-downloads" in source
    assert "--cache-only" in source
    assert "export-cache" in source
    assert '"$input_prefix/input.tar.gz"' in source
    assert "extract-archive" in source
    assert 'cmp -- "$publication/SHA256SUMS" "$roundtrip/SHA256SUMS"' in source
