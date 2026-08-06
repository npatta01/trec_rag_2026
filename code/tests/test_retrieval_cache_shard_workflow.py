from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tomllib

import pytest
import yaml

from trec_rag.hf_bucket_listing import (
    HFListingError,
    parse_hf_bucket_listing,
    require_bundle_listing,
    require_empty_listing,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / ".dstack" / "rag26-retrieval-cache-shard.yaml"
WRAPPER_PATH = REPO_ROOT / "code" / "tools" / "run_retrieval_cache_shard.sh"

IMAGE = (
    "huggingface/trl@sha256:"
    "4de10fa68e4f4e060cb41885044208e2a44715fc0d52ce83bb071b1fc6d63db1"
)
MIXEDBREAD_REVISION = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _configuration() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_dstack_shard_task_has_a_bounded_ephemeral_resource_contract() -> None:
    config = _configuration()

    assert config["type"] == "task"
    assert config["name"] == "rag26-retrieval-cache-shard"
    assert config["image"] == IMAGE
    assert config["entrypoint"] == "/bin/bash -c"
    assert config["shell"] == "bash"
    assert config["working_dir"] == "/dstack/run/trec_rag_2026"
    assert config["repos"] == [
        {
            "local_path": "..",
            "path": "/dstack/run/trec_rag_2026",
            "if_exists": "error",
        }
    ]
    assert config["commands"] == [
        "bash code/tools/run_retrieval_cache_shard.sh ${{ run.args }}"
    ]

    assert config["backends"] == ["runpod", "vastai"]
    assert config["spot_policy"] == "on-demand"
    assert config["max_price"] == 1.0
    assert config["max_duration"] == "5h"
    assert config["idle_duration"] == "0s"
    assert config["retry"] == {
        "on_events": ["no-capacity"],
        "duration": "30m",
    }

    resources = config["resources"]
    assert resources == {
        "gpu": {
            "name": ["A5000", "L4", "RTX3090", "RTX4090"],
            "count": 1,
            "memory": "24GB..",
        },
        "memory": "32GB..",
        "disk": "100GB",
    }


def test_dstack_shard_task_names_only_the_four_authorized_secrets() -> None:
    config = _configuration()
    env = config["env"]
    assert isinstance(env, list)
    secret_references: dict[str, str] = {}
    for entry in env:
        assert isinstance(entry, str)
        match = re.fullmatch(
            r"([A-Z][A-Z0-9_]*)=\$\{\{ secrets\.([A-Z][A-Z0-9_]*) \}\}",
            entry,
        )
        if match is not None:
            secret_references[match.group(1)] = match.group(2)

    expected = {
        "HF_TOKEN",
        "INDEX_URL",
        "PYSERINI_API_TOKEN",
        "OPENROUTER_API_KEY",
    }
    assert set(secret_references) == expected
    assert secret_references == {name: name for name in expected}
    assert "HF_CLI_MODE=direct" in env
    assert not any("TOKEN=" in entry and "secrets." not in entry for entry in env)


def test_wrapper_preflight_is_credential_free_and_reports_exact_nested_argv() -> None:
    before = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    env = os.environ.copy()
    for name in (
        "HF_TOKEN",
        "INDEX_URL",
        "PYSERINI_API_TOKEN",
        "OPENROUTER_API_KEY",
    ):
        env.pop(name, None)
    env["HF_CLI_MODE"] = "direct"

    result = subprocess.run(
        [
            "bash",
            str(WRAPPER_PATH),
            "--preflight",
            "--topic",
            "rag2026-0",
            "--run-id",
            "nonagentic-two-topic-20260806",
            "--config",
            "configs/rag26_competition_retrieval_v2.yaml",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "preflight=ok" in result.stdout
    assert "topic_id=rag2026-0" in result.stdout
    assert "experiment_id=nonagentic-two-topic-20260806-rag2026-0" in result.stdout
    assert (
        "remote_prefix=hf://buckets/Npatta01/trec_mlm_2026/"
        "trec_rag_2026/experiments/nonagentic-two-topic-20260806/rag2026-0"
        in result.stdout
    )
    assert (
        "runner_argv=.venv/bin/python -m trec_rag.competition_retrieval "
        "configs/local/nonagentic-two-topic-20260806-rag2026-0.yaml "
        "--topic rag2026-0" in result.stdout
    )
    assert (
        "pack_argv=.venv/bin/python -m trec_rag.competition_cache_bundle pack "
        "--config configs/local/nonagentic-two-topic-20260806-rag2026-0.yaml "
        "--topic rag2026-0" in result.stdout
    )
    assert (
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == before
    )


def test_wrapper_rejects_unsafe_identity_before_any_live_work() -> None:
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER_PATH),
            "--preflight",
            "--topic",
            "../rag2026-0",
            "--run-id",
            "unsafe/id",
            "--config",
            "configs/rag26_competition_retrieval_v2.yaml",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "safe topic ID" in result.stderr or "safe run ID" in result.stderr


def test_wrapper_encodes_model_upload_and_remote_verification_contract() -> None:
    script = WRAPPER_PATH.read_text(encoding="utf-8")

    assert MIXEDBREAD_REVISION in script
    assert MINILM_REVISION in script
    assert "local_files_only=True" in script
    assert "torch.cuda.is_available()" in script
    assert "uv sync" in script
    assert "--group cuda" in script
    assert "--locked" in script
    assert "--no-managed-python" in script
    assert "--no-python-downloads" in script
    assert 'git diff --check "$tracking_ref"' in script
    assert "git add -A" in script
    assert "git commit" in script
    assert "git status --porcelain=v1 --untracked-files=all" in script
    assert "TREC_RAG_CACHE_ROOT" in script

    archive_upload = script.index('upload_one "$bundle_archive"')
    completion_upload = script.index('upload_one "$bundle_completion"')
    remote_list = script.index(
        'remote_listing_after="$work_root/remote-listing-after.json"'
    )
    remote_download = script.index('downloaded_bundle="$work_root/downloaded-bundle"')
    remote_verify = script.index(
        'trec_rag.competition_cache_bundle verify "$downloaded_bundle"'
    )
    assert (
        archive_upload
        < completion_upload
        < remote_list
        < remote_download
        < remote_verify
    )
    assert "--delete" not in script
    assert "buckets remove" not in script
    assert "buckets delete" not in script


def test_cuda_lock_fork_supports_the_pinned_images_existing_python() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["uv"]["dependency-groups"]["cuda"] == {
        "requires-python": ">=3.11,<3.13"
    }
    cuda_dependencies = project["dependency-groups"]["cuda"]
    assert cuda_dependencies
    for requirement in cuda_dependencies:
        assert "python_version >= '3.11'" in requirement
        assert "python_version < '3.13'" in requirement
    assert (
        "[[ $python_version == 3.11 || $python_version == 3.12 ]]"
        in WRAPPER_PATH.read_text(encoding="utf-8")
    )


def test_hf_listing_parser_accepts_the_cli_empty_json_and_jsonl_forms() -> None:
    assert parse_hf_bucket_listing("") == ()
    assert parse_hf_bucket_listing(" \n") == ()
    assert parse_hf_bucket_listing("[]\n") == ()
    assert parse_hf_bucket_listing(
        '[{"path":"prefix/bundle.tar.zst","type":"file"}]\n'
    ) == ({"path": "prefix/bundle.tar.zst", "type": "file"},)
    assert parse_hf_bucket_listing(
        '{"path":"prefix/bundle.tar.zst","type":"file"}\n'
        '{"path":"prefix/bundle-complete.json","type":"file"}\n'
    ) == (
        {"path": "prefix/bundle.tar.zst", "type": "file"},
        {"path": "prefix/bundle-complete.json", "type": "file"},
    )


def test_hf_listing_requirements_fail_closed() -> None:
    require_empty_listing("")
    with pytest.raises(HFListingError, match="not empty"):
        require_empty_listing('[{"path":"prefix/existing"}]')

    complete = (
        '[{"path":"prefix/bundle.tar.zst","type":"file"},'
        '{"path":"prefix/bundle-complete.json","type":"file"}]'
    )
    require_bundle_listing(complete)
    with pytest.raises(HFListingError, match="exactly"):
        require_bundle_listing('[{"path":"prefix/bundle.tar.zst"}]')
    with pytest.raises(HFListingError, match="exactly"):
        require_bundle_listing(complete[:-1] + ',{"path":"prefix/unexpected.txt"}]')
    with pytest.raises(HFListingError, match="duplicate"):
        require_bundle_listing(
            '[{"path":"a/bundle.tar.zst"},{"path":"b/bundle.tar.zst"},'
            '{"path":"a/bundle-complete.json"}]'
        )
