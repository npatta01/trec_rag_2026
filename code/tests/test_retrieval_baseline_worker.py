from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = REPO_ROOT / "code/tools/run_retrieval_baseline_worker.sh"
LAUNCHER = REPO_ROOT / "code/tools/apply_retrieval_baseline_worker.sh"
TEMPLATE = REPO_ROOT / ".dstack/rag26-retrieval-baseline-worker.yaml"
H100_TEMPLATE = REPO_ROOT / ".dstack/rag26-retrieval-baseline-worker-h100.yaml"


def test_worker_credential_free_preflight_uses_authenticated_all_topic_bundle() -> None:
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
            "--source-revision",
            "b" * 40,
            "--source-tree",
            "c" * 40,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "preflight=ok" in completed.stdout
    assert "topic_assignment=authenticated-input-manifest" in completed.stdout
    assert "required_topic_count=119" in completed.stdout
    assert "canary_topic_ids=rag2026-1,rag2026-18" in completed.stdout
    assert f"input_manifest_sha256={'a' * 64}" in completed.stdout
    assert "live_model_processes=1" in completed.stdout
    assert f"source_revision={'b' * 40}" in completed.stdout
    assert f"source_tree={'c' * 40}" in completed.stdout


def test_worker_preflight_rejects_external_topic_selectors() -> None:
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
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "unknown argument" in completed.stderr


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
    assert value["resources"]["gpu"]["name"] == ["H200"]
    assert value["resources"]["gpu"]["memory"] == "48GB.."
    assert value["spot_policy"] == "on-demand"
    assert value["idle_duration"] == "0s"
    assert value["max_price"] == 5.0
    assert value["max_duration"] == "1h45m"

    fallback = yaml.safe_load(H100_TEMPLATE.read_text(encoding="utf-8"))
    assert fallback["resources"]["gpu"]["name"] == ["H100"]
    assert fallback["resources"]["gpu"]["count"] == 1
    assert fallback["resources"]["gpu"]["memory"] == "48GB.."
    assert fallback["max_price"] == 5.0
    assert fallback["max_duration"] == "1h45m"
    assert fallback["spot_policy"] == "on-demand"


def test_launcher_defaults_to_declined_preview_and_uses_clean_snapshot() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "mode=preview" in source
    assert "printf 'n\\n'" in source
    assert 'git clone --quiet --no-hardlinks "$REPO_ROOT" "$snapshot"' in source
    assert '[[ -z $source_status ]]' in source
    assert '"$dstack_bin" apply' in source
    assert ' -y -d -- ' in source
    assert "--input-manifest-sha256" in source
    assert "--gpu" in source
    assert "rag26-retrieval-baseline-worker-h100.yaml" in source
    assert "topic_ids" not in source
    assert 'source_tree=$(git rev-parse --verify \'HEAD^{tree}\')' in source
    assert '--source-revision "$source_head"' in source
    assert '--source-tree "$source_tree"' in source


def test_worker_accepts_exact_reviewed_tree_delivered_as_dstack_patch(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    worker = checkout / "code/tools/run_retrieval_baseline_worker.sh"
    worker.parent.mkdir(parents=True)
    shutil.copy2(WORKER, worker)
    (checkout / "transport.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"], cwd=checkout, check=True
    )
    subprocess.run(["git", "add", "-A"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=checkout, check=True)
    (checkout / "transport.txt").write_text("reviewed patch\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=checkout, check=True)
    source_tree = subprocess.run(
        ["git", "write-tree"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "reset", "-q"], cwd=checkout, check=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\necho TRANSPORT_TREE_VERIFIED >&2\nexit 17\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$*\" in\n"
        "  *os.path.realpath*) realpath \"$0\" ;;\n"
        "  *platform.python_version*) echo 3.12.0 ;;\n"
        "  *) exit 19 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            str(worker),
            "--task-name",
            "transport-test",
            "--input-prefix",
            "hf://buckets/private/test/artifacts/input",
            "--input-manifest-sha256",
            "a" * 64,
            "--output-prefix",
            "hf://buckets/private/test/experiments/output",
            "--source-revision",
            "b" * 40,
            "--source-tree",
            source_tree,
        ],
        cwd=checkout,
        env={
            **os.environ,
            "HF_TOKEN": "fixture",
            "HF_CLI_MODE": "direct",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 17
    assert "TRANSPORT_TREE_VERIFIED" in completed.stderr
    assert "dirty" not in completed.stderr


def test_worker_checks_privacy_and_output_before_model_scoring() -> None:
    source = WORKER.read_text(encoding="utf-8")

    privacy_offset = source.index('buckets info "$bucket"')
    empty_offset = source.index('require_empty_prefix "$output_prefix"')
    model_offset = source.index("snapshot_download(model, revision=revision)")
    scoring_offset = source.index("retrieval_baseline_remote_worker")
    assert privacy_offset < model_offset
    assert empty_offset < model_offset
    assert model_offset < scoring_offset
    assert "--no-managed-python" in source
    assert "--no-python-downloads" in source
    assert "replay-receipts" in source or "remote-scoring-receipt.json" in source
    assert "remote-scoring-receipt.json" in source
    assert '"$input_prefix/input.tar.gz"' in source
    assert "extract-archive" in source
    assert 'cmp -- "$publication/SHA256SUMS" "$roundtrip/SHA256SUMS"' in source
    assert "required_topic_count=119" in source


def test_worker_publishes_canary_failure_receipt_manifest_last() -> None:
    source = WORKER.read_text(encoding="utf-8")

    assert "remote-scoring-failure-receipt.json" in source
    assert "retrieval-baseline-failure-publication-v2" in source
    assert '"status": "failed"' in source
    assert "failure_receipt_sha256" in source
    assert 'buckets sync "$publication" "$output_prefix"' in source
    failure_manifest = source.index("retrieval-baseline-failure-publication-v2")
    failure_upload = source.index(
        '"$output_prefix/publication-manifest.json"', failure_manifest
    )
    assert failure_manifest < failure_upload
