from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

import pytest
import yaml

from trec_rag.facet_pilot_config import (
    load_facet_pilot_config,
    select_configured_topics,
)
from trec_rag.hf_bucket_listing import (
    HFListingError,
    parse_hf_bucket_listing,
    require_bundle_listing,
    require_empty_listing,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / ".dstack" / "rag26-retrieval-cache-shard.yaml"
DEV_CONFIG_PATH = REPO_ROOT / ".dstack" / "rag26-retrieval-cache-dev.yaml"
WRAPPER_PATH = REPO_ROOT / "code" / "tools" / "run_retrieval_cache_shard.sh"
LAUNCHER_PATH = REPO_ROOT / "code" / "tools" / "apply_retrieval_cache_shard.sh"
RAG25_CONFIG_PATH = REPO_ROOT / "configs" / "rag25_competition_retrieval_v1.yaml"
RAG26_CONFIG_PATH = REPO_ROOT / "configs" / "rag26_competition_retrieval_v2.yaml"

TRANSPORT_SENTINEL = "/dev/null/trec-rag-dstack-transport-requires-launcher"
PUBLIC_REMOTE_URL = "https://github.com/npatta01/trec_rag_2026.git"

IMAGE = (
    "huggingface/trl@sha256:"
    "4de10fa68e4f4e060cb41885044208e2a44715fc0d52ce83bb071b1fc6d63db1"
)
MIXEDBREAD_REVISION = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def test_rag25_competition_config_selects_all_22_dev_topics() -> None:
    loaded = load_facet_pilot_config(RAG25_CONFIG_PATH)
    topics = select_configured_topics(loaded)

    assert [topic.id for topic in topics] == [
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
    ]


def test_rag25_config_matches_rag26_algorithm_after_normalizing_bindings() -> None:
    rag25 = yaml.safe_load(RAG25_CONFIG_PATH.read_text(encoding="utf-8"))
    rag26 = yaml.safe_load(RAG26_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(rag25, dict)
    assert isinstance(rag26, dict)

    del rag25["experiment"]["id"]
    del rag26["experiment"]["id"]
    del rag25["topics"]["path"]
    del rag26["topics"]["path"]

    assert rag25 == rag26


def test_wrapper_preflight_accepts_numeric_rag25_topic() -> None:
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER_PATH),
            "--preflight",
            "--topic",
            "31",
            "--run-id",
            "nonagentic-rag25-dev-20260806",
            "--config",
            "configs/rag25_competition_retrieval_v1.yaml",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "topic_id=31" in result.stdout


def _run_rag25_wrapper_preflight(*topic_ids: str) -> subprocess.CompletedProcess[str]:
    args = ["bash", str(WRAPPER_PATH), "--preflight"]
    for topic_id in topic_ids:
        args.extend(["--topic", topic_id])
    args.extend(
        [
            "--run-id",
            "nonagentic-rag25-dev-20260806",
            "--config",
            "configs/rag25_competition_retrieval_v1.yaml",
        ]
    )
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_wrapper_preflight_supports_two_unique_topics() -> None:
    result = _run_rag25_wrapper_preflight("14", "37")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "topic_ids=14,37" in result.stdout
    assert "parallel_topic_processes=2" in result.stdout
    for topic_id in ("14", "37"):
        assert f"topic_id={topic_id}" in result.stdout
        assert (
            f"cache_root=/tmp/trec-rag-cache-shards/"
            f"nonagentic-rag25-dev-20260806/{topic_id}/cache"
        ) in result.stdout
        assert (
            "remote_prefix=hf://buckets/Npatta01/trec_mlm_2026/"
            "trec_rag_2026/experiments/nonagentic-rag25-dev-20260806/"
            f"{topic_id}"
        ) in result.stdout


def test_wrapper_rejects_duplicate_topic_selectors() -> None:
    result = _run_rag25_wrapper_preflight("14", "14")

    assert result.returncode != 0
    assert "--topic selectors must be unique" in result.stderr


def test_wrapper_rejects_more_than_two_topic_selectors() -> None:
    result = _run_rag25_wrapper_preflight("14", "37", "58")

    assert result.returncode != 0
    assert "one or two --topic selectors" in result.stderr


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_fake_live_wrapper(
    tmp_path: Path,
    *,
    topic_ids: tuple[str, ...],
    fail_topic: str = "",
    preexisting_topic_ids: tuple[str, ...] = (),
    force_14_144_order: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, str]:
    checkout = tmp_path / "checkout"
    wrapper = checkout / WRAPPER_PATH.relative_to(REPO_ROOT)
    source_config = checkout / RAG25_CONFIG_PATH.relative_to(REPO_ROOT)
    wrapper.parent.mkdir(parents=True)
    source_config.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER_PATH, wrapper)
    shutil.copy2(RAG25_CONFIG_PATH, source_config)

    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch", "task"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Retrieval shard test"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "retrieval-shard-test@invalid.local"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "--no-gpg-sign", "-m", "fixture"],
        cwd=checkout,
        check=True,
    )
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(remote)], check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote)],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "push", "--quiet", "--set-upstream", "origin", "task"],
        cwd=checkout,
        check=True,
    )

    git_exclude = checkout / ".git" / "info" / "exclude"
    git_exclude.write_text("/.venv/\n/configs/local/\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "remote").mkdir()
    run_id = f"fixture-{tmp_path.parent.name}-{tmp_path.name}"
    for topic_id in preexisting_topic_ids:
        topic_root = runtime / "remote" / topic_id
        topic_root.mkdir()
        (topic_root / "bundle.tar.zst").write_text(
            f"archive:{topic_id}\n", encoding="utf-8"
        )
        (topic_root / "bundle-complete.json").write_text(
            f'{{"topic_id":"{topic_id}"}}\n', encoding="utf-8"
        )
    fake_bin = runtime / "bin"
    fake_bin.mkdir()
    venv_bin = checkout / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    real_python = REPO_ROOT / ".venv" / "bin" / "python"

    _write_executable(
        venv_bin / "python",
        f"#!{real_python}\n"
        + r'''from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import yaml


runtime = Path(os.environ["FAKE_SHARD_RUNTIME"])


def event(*parts: object) -> None:
    line = "|".join(str(part).replace("\n", "\\n") for part in parts) + "\n"
    descriptor = os.open(runtime / "events.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)


args = sys.argv[1:]
if args and args[0] == "-":
    if len(args) == 5:
        _, source, destination, experiment_id, topic_id = args
        value = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        value["experiment"]["id"] = experiment_id
        value.setdefault("execution", {})["topic_workers"] = 1
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        event("config", topic_id, target, experiment_id, 1)
        raise SystemExit(0)
    if len(args) == 3 and Path(args[1]).is_file():
        event("validate-topic", args[2])
        raise SystemExit(0)
    if len(args) == 3:
        event("model-prefetch", os.environ.get("HF_HOME", ""), args[1], args[2])
        raise SystemExit(0)
    if len(args) == 2:
        event("bucket-private-check", args[1])
        raise SystemExit(0)
    if len(args) == 1:
        event("cuda-check")
        raise SystemExit(0)

if args[:2] == ["-m", "trec_rag.hf_bucket_listing"]:
    event("listing-check", *args[2:])
    from trec_rag.hf_bucket_listing import main

    raise SystemExit(main(args[2:]))

if args[:2] == ["-m", "trec_rag.competition_retrieval"]:
    rest = args[2:]
    topic_id = rest[rest.index("--topic") + 1]
    config_path = Path(rest[0])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    event(
        "retrieval-start",
        topic_id,
        os.environ.get("TREC_RAG_CACHE_ROOT", ""),
        config["experiment"]["id"],
    )
    started = runtime / "started"
    started.mkdir(exist_ok=True)
    (started / topic_id).touch()
    expected = int(os.environ["FAKE_EXPECTED_TOPICS"])
    deadline = time.monotonic() + 2.0
    while len(tuple(started.iterdir())) < expected:
        if time.monotonic() >= deadline:
            event("retrieval-timeout", topic_id)
            raise SystemExit(91)
        time.sleep(0.01)
    if topic_id == os.environ.get("FAKE_FAIL_TOPIC", ""):
        event("retrieval-failed", topic_id)
        raise SystemExit(42)
    event("retrieval-complete", topic_id)
    raise SystemExit(0)

if args[:2] == ["-m", "trec_rag.competition_cache_bundle"]:
    rest = args[2:]
    command = rest[0]
    if command == "pack":
        topic_id = rest[rest.index("--topic") + 1]
        destination = Path(rest[rest.index("--destination") + 1])
        destination.mkdir(parents=True)
        (destination / "bundle.tar.zst").write_text(
            f"archive:{topic_id}\n", encoding="utf-8"
        )
        (destination / "bundle-complete.json").write_text(
            f'{{"topic_id":"{topic_id}"}}\n', encoding="utf-8"
        )
        event(
            "pack",
            topic_id,
            os.environ.get("TREC_RAG_CACHE_ROOT", ""),
            destination,
        )
        raise SystemExit(0)
    if command == "verify":
        destination = Path(rest[1])
        required = {"bundle.tar.zst", "bundle-complete.json"}
        if not destination.is_dir() or {path.name for path in destination.iterdir()} != required:
            raise SystemExit(92)
        event("verify", destination)
        raise SystemExit(0)

event("unexpected-python", *args)
raise SystemExit(93)
''',
    )
    _write_executable(
        venv_bin / "hf",
        f"#!{real_python}\n"
        + r'''from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time


runtime = Path(os.environ["FAKE_SHARD_RUNTIME"])
remote_root = runtime / "remote"


def event(*parts: object) -> None:
    line = "|".join(str(part).replace("\n", "\\n") for part in parts) + "\n"
    descriptor = os.open(runtime / "events.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)


args = sys.argv[1:]
if args == ["buckets", "--help"]:
    event("hf-help")
    raise SystemExit(0)
if args[:2] == ["auth", "whoami"]:
    event("hf-auth")
    print('{"name":"fixture"}')
    raise SystemExit(0)
if args[:2] == ["buckets", "info"]:
    event("hf-bucket-info")
    print('{"private":true}')
    raise SystemExit(0)
if args[:2] == ["buckets", "list"]:
    prefix = args[2]
    marker = "hf://buckets/Npatta01/trec_mlm_2026/"
    if not prefix.startswith(marker):
        raise SystemExit(97)
    requested_path = prefix.removeprefix(marker).rstrip("/")
    run_path = os.environ["FAKE_RUN_BUCKET_PATH"]
    cache_root = os.environ.get("TREC_RAG_CACHE_ROOT", "")
    caller_topic = Path(cache_root).parent.name if cache_root else ""
    if os.environ.get("FAKE_FORCE_14_144_ORDER") == "1" and caller_topic == "14":
        sibling_root = remote_root / "144"
        deadline = time.monotonic() + 3.0
        while not sibling_root.is_dir() or len(tuple(sibling_root.iterdir())) < 2:
            if time.monotonic() >= deadline:
                event("hf-list-timeout", requested_path, caller_topic)
                raise SystemExit(98)
            time.sleep(0.01)
    all_entries = []
    for topic_root in sorted(remote_root.iterdir()):
        if not topic_root.is_dir():
            continue
        topic_path = f"{run_path}/{topic_root.name}"
        all_entries.append({"path": topic_path, "type": "directory"})
        all_entries.extend(
            {"path": f"{topic_path}/{path.name}", "type": "file"}
            for path in sorted(topic_root.iterdir())
        )
    entries = [
        entry
        for entry in all_entries
        if entry["path"] != requested_path
        and entry["path"].startswith(requested_path)
    ]
    event("hf-list", requested_path, caller_topic, len(entries))
    print(json.dumps(entries))
    raise SystemExit(0)
if args[:2] == ["buckets", "sync"]:
    source = Path(args[2])
    prefix = args[3]
    topic_id = prefix.rstrip("/").split("/")[-1]
    topic_root = remote_root / topic_id
    topic_root.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        target = topic_root / path.name
        if target.exists():
            raise SystemExit(94)
        shutil.copyfile(path, target)
        event("upload", topic_id, path.name)
    raise SystemExit(0)
if args[:2] == ["buckets", "cp"]:
    source = args[2]
    destination = Path(args[3])
    pieces = source.rstrip("/").split("/")
    topic_id, basename = pieces[-2:]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(remote_root / topic_id / basename, destination)
    event("remote-copy", topic_id, basename)
    raise SystemExit(0)

event("unexpected-hf", *args)
raise SystemExit(95)
''',
    )
    _write_executable(
        fake_bin / "python3",
        """#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} == -c ]] || exit 96
printf '3.12\\n'
""",
    )
    _write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'uv-sync\\n' >>"$FAKE_SHARD_RUNTIME/events.log"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "HF_CLI_MODE": "direct",
            "HF_TOKEN": "fixture",
            "INDEX_URL": "https://index.invalid",
            "PYSERINI_API_TOKEN": "fixture",
            "OPENROUTER_API_KEY": "fixture",
            "FAKE_SHARD_RUNTIME": str(runtime),
            "FAKE_EXPECTED_TOPICS": str(len(topic_ids)),
            "FAKE_FAIL_TOPIC": fail_topic,
            "FAKE_FORCE_14_144_ORDER": "1" if force_14_144_order else "0",
            "FAKE_RUN_BUCKET_PATH": (
                f"trec_rag_2026/experiments/{run_id}"
            ),
        }
    )
    args = ["bash", str(wrapper)]
    for topic_id in topic_ids:
        args.extend(["--topic", topic_id])
    args.extend(
        [
            "--run-id",
            run_id,
            "--config",
            "configs/rag25_competition_retrieval_v1.yaml",
        ]
    )
    result = subprocess.run(
        args,
        cwd=checkout,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result, checkout, runtime, run_id


def _event_rows(runtime: Path) -> list[list[str]]:
    return [
        line.split("|")
        for line in (runtime / "events.log").read_text(encoding="utf-8").splitlines()
    ]


def _published_bundle_names(runtime: Path, topic_id: str) -> set[str]:
    topic_root = runtime / "remote" / topic_id
    if not topic_root.exists():
        return set()
    return {path.name for path in topic_root.iterdir()}


def test_wrapper_live_single_topic_behavior_is_preserved(tmp_path: Path) -> None:
    result, _, runtime, _ = _run_fake_live_wrapper(tmp_path, topic_ids=("31",))

    assert result.returncode == 0, result.stderr
    assert "shard_status=complete" in result.stdout
    assert "topic_id=31" in result.stdout
    assert _published_bundle_names(runtime, "31") == {
        "bundle.tar.zst",
        "bundle-complete.json",
    }


def test_wrapper_topic_14_ignores_a_preexisting_144_sibling(tmp_path: Path) -> None:
    result, _, runtime, run_id = _run_fake_live_wrapper(
        tmp_path,
        topic_ids=("14",),
        preexisting_topic_ids=("144",),
    )

    assert result.returncode == 0, result.stderr
    assert _published_bundle_names(runtime, "14") == {
        "bundle.tar.zst",
        "bundle-complete.json",
    }
    assert _published_bundle_names(runtime, "144") == {
        "bundle.tar.zst",
        "bundle-complete.json",
    }
    expected_run_path = f"trec_rag_2026/experiments/{run_id}"
    assert {
        row[1] for row in _event_rows(runtime) if row[0] == "hf-list"
    } == {expected_run_path}


def test_wrapper_concurrent_topics_14_and_144_publish_exact_subtrees(
    tmp_path: Path,
) -> None:
    result, _, runtime, run_id = _run_fake_live_wrapper(
        tmp_path,
        topic_ids=("14", "144"),
        force_14_144_order=True,
    )

    assert result.returncode == 0, result.stderr
    for topic_id in ("14", "144"):
        assert _published_bundle_names(runtime, topic_id) == {
            "bundle.tar.zst",
            "bundle-complete.json",
        }
    rows = _event_rows(runtime)
    expected_run_path = f"trec_rag_2026/experiments/{run_id}"
    assert {row[1] for row in rows if row[0] == "hf-list"} == {
        expected_run_path
    }
    assert any(
        row[:4] == ["hf-list", expected_run_path, "14", "3"] for row in rows
    )


def test_wrapper_runs_two_isolated_parallel_topic_jobs_after_shared_setup(
    tmp_path: Path,
) -> None:
    result, checkout, runtime, run_id = _run_fake_live_wrapper(
        tmp_path,
        topic_ids=("14", "37"),
    )

    assert result.returncode == 0, result.stderr
    rows = _event_rows(runtime)
    assert sum(row[0] == "uv-sync" for row in rows) == 1
    assert sum(row[0] == "hf-auth" for row in rows) == 1
    assert sum(row[0] == "hf-bucket-info" for row in rows) == 1
    assert sum(row[0] == "model-prefetch" for row in rows) == 1
    assert sum(row[0] == "cuda-check" for row in rows) == 1
    assert {row[1] for row in rows if row[0] == "retrieval-start"} == {"14", "37"}
    assert not any(row[0] == "retrieval-timeout" for row in rows)

    for topic_id in ("14", "37"):
        expected_experiment = f"{run_id}-{topic_id}"
        expected_cache = f"/tmp/trec-rag-cache-shards/{run_id}/{topic_id}/cache"
        expected_work = f"/tmp/trec-rag-cache-shards/{run_id}/{topic_id}"
        config_path = checkout / "configs" / "local" / f"{expected_experiment}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["experiment"]["id"] == expected_experiment
        assert config["execution"]["topic_workers"] == 1
        assert any(
            row[:4] == ["retrieval-start", topic_id, expected_cache, expected_experiment]
            for row in rows
        )
        assert any(
            row == ["pack", topic_id, expected_cache, f"{expected_work}/bundle"]
            for row in rows
        )
        assert {
            row[1]
            for row in rows
            if row[0] == "verify" and row[1].startswith(expected_work)
        } == {
            f"{expected_work}/bundle",
            f"{expected_work}/downloaded-bundle",
        }
        assert [row[2] for row in rows if row[:2] == ["upload", topic_id]] == [
            "bundle.tar.zst",
            "bundle-complete.json",
        ]
        assert _published_bundle_names(runtime, topic_id) == {
            "bundle.tar.zst",
            "bundle-complete.json",
        }


def test_failed_topic_does_not_prevent_successful_sibling_publication(
    tmp_path: Path,
) -> None:
    result, _, runtime, _ = _run_fake_live_wrapper(
        tmp_path,
        topic_ids=("14", "37"),
        fail_topic="14",
    )

    assert result.returncode != 0
    assert "topic shard failed: 14" in result.stderr
    rows = _event_rows(runtime)
    assert {row[1] for row in rows if row[0] == "retrieval-start"} == {"14", "37"}
    assert _published_bundle_names(runtime, "14") == set()
    assert _published_bundle_names(runtime, "37") == {
        "bundle.tar.zst",
        "bundle-complete.json",
    }
    assert [row[2] for row in rows if row[:2] == ["upload", "37"]] == [
        "bundle.tar.zst",
        "bundle-complete.json",
    ]


def _configuration() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    "script_path",
    [WRAPPER_PATH, LAUNCHER_PATH],
    ids=["remote-wrapper", "local-launcher"],
)
def test_shard_tool_help_describes_shared_safe_configured_topic_ids(
    script_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", str(script_path), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "--topic SAFE_ID" in result.stdout
    assert "[--topic SAFE_ID]" in result.stdout
    assert "rag2026-N" not in result.stdout
    assert "RAG25 topic 31" in result.stdout
    assert "RAG26 topic rag2026-0" in result.stdout
    assert "safe topic ID" in result.stdout
    assert "present in the configured topics" in result.stdout


def test_dstack_dev_environment_is_interactive_bounded_and_48gb_only() -> None:
    value = yaml.safe_load(DEV_CONFIG_PATH.read_text(encoding="utf-8"))
    assert value["type"] == "dev-environment"
    assert value["ide"] == "vscode"
    assert value["working_dir"] == "/workflow"
    assert value["image"] == IMAGE
    assert value["resources"] == {
        "gpu": {"name": ["A40", "A6000", "L40S"], "count": 1, "memory": "48GB.."},
        "memory": "32GB..",
        "disk": "100GB",
    }
    assert value["spot_policy"] == "on-demand"
    assert value["max_price"] == 1.0
    assert value["max_duration"] == "5h"
    assert value["idle_duration"] == "30m"
    assert "HF_TOKEN=${{ secrets.hf_token }}" in value["env"]


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
            "local_path": TRANSPORT_SENTINEL,
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
            "name": ["A40", "A6000", "L40S"],
            "count": 1,
            "memory": "48GB..",
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
            r"([A-Z][A-Z0-9_]*)=\$\{\{ secrets\.([A-Za-z][A-Za-z0-9_]*) \}\}",
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
    assert secret_references == {
        "HF_TOKEN": "hf_token",
        "INDEX_URL": "INDEX_URL",
        "PYSERINI_API_TOKEN": "PYSERINI_API_TOKEN",
        "OPENROUTER_API_KEY": "OPENROUTER_API_KEY",
    }
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


def test_wrapper_preflight_rejects_a_topic_absent_from_the_config() -> None:
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER_PATH),
            "--preflight",
            "--topic",
            "rag2026-999999",
            "--run-id",
            "nonagentic-two-topic-20260806",
            "--config",
            "configs/rag26_competition_retrieval_v2.yaml",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "absent from the configured topics" in result.stderr


def test_wrapper_preflight_rejects_nonignored_untracked_transport_files() -> None:
    candidate = REPO_ROOT / ".dstack-preflight-untracked-test"
    candidate.write_text("transport audit fixture\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [
                "bash",
                str(WRAPPER_PATH),
                "--preflight",
                "--topic",
                "rag2026-0",
                "--run-id",
                "nonagentic-two-topic-20260806",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        candidate.unlink(missing_ok=True)

    assert result.returncode != 0
    assert "non-ignored untracked files" in result.stderr
    assert candidate.name in result.stderr


@pytest.mark.parametrize(
    "config_arg",
    [
        str(REPO_ROOT / "configs" / "rag26_competition_retrieval_v2.yaml"),
        "configs/local/ignored-retrieval-config.yaml",
    ],
)
def test_wrapper_rejects_configs_that_cannot_enter_the_committed_snapshot(
    config_arg: str,
) -> None:
    ignored_config = REPO_ROOT / "configs" / "local" / "ignored-retrieval-config.yaml"
    if config_arg.startswith("configs/local/"):
        ignored_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            REPO_ROOT / "configs" / "rag26_competition_retrieval_v2.yaml",
            ignored_config,
        )
    try:
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
                config_arg,
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        ignored_config.unlink(missing_ok=True)

    assert result.returncode != 0
    assert "tracked, repository-relative config" in result.stderr


def test_wrapper_preflight_uses_only_the_locked_project_hf_cli(tmp_path: Path) -> None:
    restricted_bin = tmp_path / "bin"
    restricted_bin.mkdir()
    for command in ("bash", "git", "python3", "uv"):
        executable = shutil.which(command)
        assert executable is not None
        (restricted_bin / command).symlink_to(executable)

    env = os.environ.copy()
    env["PATH"] = f"{restricted_bin}:/usr/bin:/bin"
    env["HF_CLI_MODE"] = "direct"
    result = subprocess.run(
        [
            str(restricted_bin / "bash"),
            str(WRAPPER_PATH),
            "--preflight",
            "--topic",
            "rag2026-0",
            "--run-id",
            "nonagentic-two-topic-20260806",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "preflight=ok" in result.stdout


def test_wrapper_rejects_unlocked_uvx_hf_mode() -> None:
    env = os.environ.copy()
    env["HF_CLI_MODE"] = "uvx"
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER_PATH),
            "--preflight",
            "--topic",
            "rag2026-0",
            "--run-id",
            "nonagentic-two-topic-20260806",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "HF_CLI_MODE must be direct" in result.stderr


def _clean_launcher_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(checkout)],
        check=True,
    )
    for source in (CONFIG_PATH, WRAPPER_PATH, LAUNCHER_PATH):
        relative = source.relative_to(REPO_ROOT)
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (checkout / ".dstack" / "test-transport-marker").write_text(
        "test-only committed transport marker\n", encoding="utf-8"
    )
    (checkout / ".git" / "info" / "exclude").write_text("/.venv\n", encoding="utf-8")
    (checkout / ".venv").symlink_to(REPO_ROOT / ".venv", target_is_directory=True)
    for name in ("trec-rag-data", "trec-rag-skills", "ragdoll"):
        subprocess.run(
            ["git", "config", f"submodule.{name}.url", str(REPO_ROOT / name)],
            cwd=checkout,
            check=True,
        )
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "add", ".dstack", "code/tools"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Retrieval shard test",
            "-c",
            "user.email=retrieval-shard-test@invalid.local",
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "test sanitized dstack transport",
        ],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "remote",
            "set-url",
            "origin",
            PUBLIC_REMOTE_URL,
        ],
        cwd=checkout,
        check=True,
    )
    return checkout


def _fake_dstack(
    tmp_path: Path, *, mutate_template_after_clone: Path | None = None
) -> tuple[Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    observed = tmp_path / "dstack-observed.txt"
    fake = fake_bin / "dstack"
    fake.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --version ]]; then
  printf '0.20.29\\n'
  exit 0
fi
[[ ${1:-} == apply ]]
[[ ${2:-} == -f ]]
task_config=$3
shift 3
"$DSTACK_ASSERT_PYTHON" - "$task_config" <<'PY'
from pathlib import Path
import os
import subprocess
import sys
import yaml

task_config = Path(sys.argv[1]).resolve()
assert Path.cwd() == task_config.parent
config = yaml.safe_load(task_config.read_text(encoding="utf-8"))
snapshot = Path(config["repos"][0]["local_path"])
assert snapshot.is_absolute() and snapshot.is_dir()
assert not (snapshot / ".env").exists()
assert not (snapshot / ".env.local").exists()
assert subprocess.run(
    ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    cwd=snapshot,
    check=True,
    capture_output=True,
    text=True,
).stdout == ""
assert subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=snapshot, check=True,
    capture_output=True, text=True,
).stdout.strip() == os.environ["EXPECTED_TRANSPORT_HEAD"]
assert subprocess.run(
    ["git", "rev-parse", "@{upstream}"], cwd=snapshot, check=True,
    capture_output=True, text=True,
).stdout.strip() == os.environ["EXPECTED_TRANSPORT_BASE"]
assert subprocess.run(
    ["git", "remote", "get-url", "origin"], cwd=snapshot, check=True,
    capture_output=True, text=True,
).stdout.strip() == "https://github.com/npatta01/trec_rag_2026.git"
assert config["commands"] == [
    "bash code/tools/run_retrieval_cache_shard.sh ${{ run.args }}"
]
PY
printf '%s\\n' "$*" >"$DSTACK_OBSERVED"
printf 'FAKE DSTACK PREVIEW\\n'
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    if mutate_template_after_clone is not None:
        fake_git = fake_bin / "git"
        real_git = shutil.which("git")
        assert real_git is not None
        fake_git.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == clone ]]; then
  "$REAL_GIT" "$@"
  printf '\ncommands:\n  - "printf unsafe-live-template"\n' >>"$MUTATE_TEMPLATE"
  exit 0
fi
exec "$REAL_GIT" "$@"
""",
            encoding="utf-8",
        )
        fake_git.chmod(0o755)
    return fake_bin, observed


def test_launcher_couples_apply_to_a_clean_committed_only_snapshot(
    tmp_path: Path,
) -> None:
    checkout = _clean_launcher_checkout(tmp_path)
    fake_bin, observed = _fake_dstack(
        tmp_path,
        mutate_template_after_clone=(checkout / CONFIG_PATH.relative_to(REPO_ROOT)),
    )
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_base = subprocess.run(
        ["git", "rev-parse", "@{upstream}"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["DSTACK_ASSERT_PYTHON"] = str(REPO_ROOT / ".venv" / "bin" / "python")
    env["DSTACK_OBSERVED"] = str(observed)
    env["EXPECTED_TRANSPORT_HEAD"] = expected_head
    env["EXPECTED_TRANSPORT_BASE"] = expected_base
    env["REAL_GIT"] = shutil.which("git") or ""
    env["MUTATE_TEMPLATE"] = str(checkout / CONFIG_PATH.relative_to(REPO_ROOT))

    result = subprocess.run(
        [
            "bash",
            str(checkout / LAUNCHER_PATH.relative_to(REPO_ROOT)),
            "--preview",
            "--name",
            "rag26-cache-rag2026-0",
            "--",
            "--topic",
            "rag2026-0",
            "--run-id",
            "nonagentic-two-topic-20260806",
        ],
        cwd=checkout,
        env=env,
        input="n\n",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "FAKE DSTACK PREVIEW\n"
    assert result.stderr == ""
    assert observed.read_text(encoding="utf-8").startswith(
        "-n rag26-cache-rag2026-0 -- --topic rag2026-0 "
        "--run-id nonagentic-two-topic-20260806"
    )


def test_launcher_rejects_a_dirty_source_before_dstack(tmp_path: Path) -> None:
    checkout = _clean_launcher_checkout(tmp_path)
    fake_bin, observed = _fake_dstack(tmp_path)
    (checkout / "untracked-secret.txt").write_text(
        "must not travel\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(checkout / LAUNCHER_PATH.relative_to(REPO_ROOT)),
            "--preview",
            "--name",
            "rag26-cache-rag2026-0",
            "--",
            "--topic",
            "rag2026-0",
            "--run-id",
            "nonagentic-two-topic-20260806",
        ],
        cwd=checkout,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "source worktree must be clean" in result.stderr
    assert not observed.exists()


@pytest.mark.parametrize(
    "remote_url",
    [
        "https://user:password@github.com/npatta01/trec_rag_2026.git",
        "ssh://git:password@github.com/npatta01/trec_rag_2026.git",
        "https://github.com/npatta01/trec_rag_2026.git?token=secret",
        "/tmp/local-repository",
    ],
)
def test_launcher_rejects_noncanonical_or_credential_bearing_remote_urls(
    tmp_path: Path,
    remote_url: str,
) -> None:
    checkout = _clean_launcher_checkout(tmp_path)
    subprocess.run(
        ["git", "remote", "set-url", "origin", remote_url],
        cwd=checkout,
        check=True,
    )

    result = subprocess.run(
        [
            "bash",
            str(checkout / LAUNCHER_PATH.relative_to(REPO_ROOT)),
            "--preview",
            "--name",
            "rag26-cache-rag2026-0",
            "--",
            "--topic",
            "rag2026-0",
            "--run-id",
            "nonagentic-two-topic-20260806",
        ],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "public GitHub repository without embedded credentials" in result.stderr


def test_dstack_transport_ignores_both_supported_secret_env_files() -> None:
    for name in (".env", ".env.local"):
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", name],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, f"{name} can enter dstack repo transport"


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
    assert "safe configured topic ID" in result.stderr or "safe run ID" in result.stderr


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
    assert 'git --no-pager diff --check "$tracking_ref"' in script
    assert 'git diff --check "$tracking_ref"' not in script
    assert "git add -A" in script
    assert "git commit" in script
    assert "git status --porcelain=v1 --untracked-files=all" in script
    assert "TREC_RAG_CACHE_ROOT" in script
    assert script.index("git add -A") < script.index("source_config_rel=$(git ls-files")

    uv_sync = script.index("uv sync")
    locked_hf = script.index(
        '[[ -x $venv_hf ]] || die "the locked project environment did not install hf"'
    )
    hf_auth = script.index("hf_cli auth whoami")
    prefix_guard = script.index("require-empty")
    model_download = script.index("snapshot_download")
    retrieval_run = script.index('"$venv_python" -m trec_rag.competition_retrieval')
    assert uv_sync < locked_hf < hf_auth < prefix_guard < model_download < retrieval_run

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


_HF_RUN_PREFIX = "trec_rag_2026/experiments/fixture-run"
_HF_TOPIC_14_PREFIX = f"{_HF_RUN_PREFIX}/14"


def test_hf_listing_exact_topic_boundary_keeps_14_empty_when_144_exists() -> None:
    sibling_listing = json.dumps(
        [
            {"path": f"{_HF_RUN_PREFIX}/144", "type": "directory"},
            {
                "path": f"{_HF_RUN_PREFIX}/144/bundle.tar.zst",
                "type": "file",
            },
            {
                "path": f"{_HF_RUN_PREFIX}/144/bundle-complete.json",
                "type": "file",
            },
        ]
    )

    require_empty_listing(sibling_listing, topic_prefix=_HF_TOPIC_14_PREFIX)
    with pytest.raises(HFListingError, match="not empty"):
        require_empty_listing(
            json.dumps(
                [
                    {"path": f"{_HF_RUN_PREFIX}/144", "type": "directory"},
                    {
                        "path": f"{_HF_TOPIC_14_PREFIX}/existing",
                        "type": "file",
                    },
                ]
            ),
            topic_prefix=_HF_TOPIC_14_PREFIX,
        )


def test_hf_listing_bundle_requires_exact_two_files_only_in_target_subtree() -> None:
    complete_records = [
        {"path": _HF_TOPIC_14_PREFIX, "type": "directory"},
        {
            "path": f"{_HF_TOPIC_14_PREFIX}/bundle.tar.zst",
            "type": "file",
        },
        {
            "path": f"{_HF_TOPIC_14_PREFIX}/bundle-complete.json",
            "type": "file",
        },
        {"path": f"{_HF_RUN_PREFIX}/144", "type": "directory"},
        {
            "path": f"{_HF_RUN_PREFIX}/144/bundle.tar.zst",
            "type": "file",
        },
        {
            "path": f"{_HF_RUN_PREFIX}/144/bundle-complete.json",
            "type": "file",
        },
    ]
    complete = json.dumps(complete_records)

    require_bundle_listing(complete, topic_prefix=_HF_TOPIC_14_PREFIX)
    with pytest.raises(HFListingError, match="exactly"):
        require_bundle_listing(
            json.dumps(
                [
                    {
                        "path": f"{_HF_TOPIC_14_PREFIX}/bundle.tar.zst",
                        "type": "file",
                    }
                ]
            ),
            topic_prefix=_HF_TOPIC_14_PREFIX,
        )
    with pytest.raises(HFListingError, match="exactly"):
        require_bundle_listing(
            json.dumps(
                [
                    *complete_records,
                    {
                        "path": f"{_HF_TOPIC_14_PREFIX}/unexpected.txt",
                        "type": "file",
                    },
                ]
            ),
            topic_prefix=_HF_TOPIC_14_PREFIX,
        )
    with pytest.raises(HFListingError, match="duplicate"):
        require_bundle_listing(
            json.dumps(
                [
                    {
                        "path": f"{_HF_TOPIC_14_PREFIX}/bundle.tar.zst",
                        "type": "file",
                    },
                    {
                        "path": f"{_HF_TOPIC_14_PREFIX}/bundle.tar.zst",
                        "type": "file",
                    },
                    {
                        "path": f"{_HF_TOPIC_14_PREFIX}/bundle-complete.json",
                        "type": "file",
                    },
                ]
            ),
            topic_prefix=_HF_TOPIC_14_PREFIX,
        )


@pytest.mark.parametrize(
    ("path", "entry_type"),
    [
        ("trec_rag_2026/experiments/other-run/14/file", "file"),
        (f"{_HF_RUN_PREFIX}/../fixture-run/14/file", "file"),
        (f"{_HF_RUN_PREFIX}\\14\\file", "file"),
        (f"/{_HF_TOPIC_14_PREFIX}/file", "file"),
        (f"{_HF_RUN_PREFIX}/14.bad/file", "file"),
        (f"{_HF_RUN_PREFIX}/14bundle", "file"),
        (f"{_HF_RUN_PREFIX}/144", "file"),
        (f"{_HF_RUN_PREFIX}/144/file", "symlink"),
    ],
    ids=(
        "outside-run-parent",
        "traversal",
        "backslash",
        "absolute",
        "unsafe-lookalike-topic",
        "lookalike-root-file",
        "sibling-root-file",
        "unknown-entry-type",
    ),
)
def test_hf_listing_rejects_malformed_or_non_subtree_records(
    path: str,
    entry_type: str,
) -> None:
    listing = json.dumps([{"path": path, "type": entry_type}])

    with pytest.raises(HFListingError):
        require_empty_listing(listing, topic_prefix=_HF_TOPIC_14_PREFIX)
