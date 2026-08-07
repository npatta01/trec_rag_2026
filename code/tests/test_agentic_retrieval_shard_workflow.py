from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / ".dstack" / "rag26-agentic-retrieval-worker.yaml"
WRAPPER_PATH = REPO_ROOT / "code" / "tools" / "run_agentic_retrieval_worker.sh"
LAUNCHER_PATH = REPO_ROOT / "code" / "tools" / "apply_agentic_retrieval_worker.sh"
AGENTIC_CONFIG = REPO_ROOT / "configs" / "rag26_competition_agentic_retrieval_v1.yaml"
IMAGE = (
    "huggingface/trl@sha256:"
    "4de10fa68e4f4e060cb41885044208e2a44715fc0d52ce83bb071b1fc6d63db1"
)


def _run(script: Path, *args: str, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", str(script), *args], cwd=cwd, env=env, check=False,
        capture_output=True, text=True,
    )


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_agentic_dstack_task_is_pinned_and_single_gpu_bounded() -> None:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert value["type"] == "task"
    assert value["name"] == "rag26-agentic-retrieval-worker"
    assert value["image"] == IMAGE
    assert value["entrypoint"] == "/bin/bash -c"
    assert value["shell"] == "bash"
    assert value["working_dir"] == "/dstack/run/trec_rag_2026"
    assert value["resources"] == {
        "gpu": {"name": ["A40", "A6000", "L40S"], "count": 1, "memory": "48GB.."},
        "memory": "48GB..",
        "disk": "100GB",
    }
    assert value["backends"] == ["runpod", "vastai"]
    assert value["spot_policy"] == "on-demand"
    assert value["max_price"] == 1.0
    assert value["max_duration"] == "6h"
    assert value["idle_duration"] == "0s"
    assert value["retry"] == {"on_events": ["no-capacity"], "duration": "30m"}


def test_agentic_task_has_only_required_secret_bindings_and_private_transport() -> None:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert value["repos"] == [{
        "local_path": "/dev/null/trec-rag-dstack-agentic-transport-requires-launcher",
        "path": "/dstack/run/trec_rag_2026",
        "if_exists": "error",
    }]
    assert value["env"] == [
        "INDEX_URL=${{ secrets.INDEX_URL }}",
        "PYSERINI_API_TOKEN=${{ secrets.PYSERINI_API_TOKEN }}",
        "OPENROUTER_API_KEY=${{ secrets.OPENROUTER_API_KEY }}",
        "HF_TOKEN=${{ secrets.hf_token }}",
        "HF_CLI_MODE=direct",
    ]
    assert value["commands"] == [
        "bash code/tools/run_agentic_retrieval_worker.sh ${{ run.args }}"
    ]
    serialized = CONFIG_PATH.read_text(encoding="utf-8")
    assert ".env" not in serialized
    assert ".env.local" not in serialized


@pytest.mark.parametrize("script", [WRAPPER_PATH, LAUNCHER_PATH])
def test_tools_default_to_preview_safe_contract(script: Path) -> None:
    result = _run(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "--task-name" in result.stdout
    assert "--run-id" in result.stdout
    assert "--plan-sha256" in result.stdout
    assert "--artifact-prefix" in result.stdout
    assert "--topic" in result.stdout
    assert "--launch" in result.stdout


def test_launcher_never_creates_transport_commit_or_mutates_checked_in_template() -> None:
    body = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "git commit" not in body
    assert "--launch" in body
    assert "--preview" in body
    assert "plan_sha256" in body
    assert "artifact_prefix" in body
    assert ".env.local" in body


def test_wrapper_preflight_validates_exact_plan_digest_and_unique_topics(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "code/tools").mkdir(parents=True)
    (checkout / "configs").mkdir()
    (checkout / "outputs").mkdir()
    shutil.copy2(WRAPPER_PATH, checkout / "code/tools/run_agentic_retrieval_worker.sh")
    config = checkout / "configs/agentic.yaml"
    config.write_text(AGENTIC_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    plan = checkout / "plan.json"
    plan.write_text(json.dumps({"plan_sha256": "a" * 64, "run_id": "run-0", "planned_topic_ids": ["rag2026-0", "rag2026-1"]}), encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", "--initial-branch", "task"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "--quiet", "--no-gpg-sign", "-m", "fixture"], cwd=checkout, check=True)
    result = _run(
        checkout / "code/tools/run_agentic_retrieval_worker.sh",
        "--preflight", "--task-name", "task-0", "--run-id", "run-0",
        "--plan", str(plan), "--plan-sha256", "a" * 64,
        "--config", str(config), "--artifact-prefix", "hf://buckets/private/trec_rag_2026/experiments/run-0",
        "--topic", "rag2026-0", "--topic", "rag2026-0",
        cwd=checkout,
    )
    assert result.returncode != 0
    assert "unique" in result.stderr or "unknown argument" in result.stderr

    result = _run(
        checkout / "code/tools/run_agentic_retrieval_worker.sh",
        "--preflight", "--task-name", "task-0", "--run-id", "run-0",
        "--plan", str(plan), "--plan-sha256", "b" * 64,
        "--config", str(config), "--artifact-prefix", "hf://buckets/private/trec_rag_2026/experiments/run-0",
        "--topic", "rag2026-0", cwd=checkout,
    )
    assert result.returncode != 0
    assert "plan" in result.stderr.lower()


def _fake_agentic_checkout(tmp_path: Path, *, fail_topic: str = "") -> tuple[Path, Path, dict[str, str]]:
    checkout = tmp_path / "checkout"
    (checkout / "code/tools").mkdir(parents=True)
    (checkout / "configs").mkdir()
    shutil.copy2(WRAPPER_PATH, checkout / "code/tools/run_agentic_retrieval_worker.sh")
    config = checkout / "configs/agentic.yaml"
    config.write_text(AGENTIC_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    plan = checkout / "plan.json"
    plan.write_text(json.dumps({"plan_sha256": "a" * 64, "run_id": "run-0", "planned_topic_ids": ["rag2026-0", "rag2026-1"]}), encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", "--initial-branch", "task"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "--quiet", "--no-gpg-sign", "-m", "fixture"], cwd=checkout, check=True)
    runtime = tmp_path / "runtime"
    (runtime / "remote").mkdir(parents=True)
    (runtime / "bin").mkdir()
    venv = checkout / ".venv/bin"
    venv.mkdir(parents=True)
    real_python = shutil.which("python3")
    assert real_python
    _write_executable(
        venv / "python",
        f"#!{real_python}\n"
        + """import os
import pathlib
import sys

runtime = pathlib.Path(os.environ["FAKE_RUNTIME"])
args = sys.argv[1:]
if args[:2] == ["-m", "trec_rag.competition_agentic_worker"]:
    topic = args[args.index("--topic") + 1]
    events = runtime / "events"
    events.mkdir(exist_ok=True)
    with (events / "worker.log").open("a") as stream:
        stream.write("worker|" + topic + "\\n")
    if topic == os.environ.get("FAKE_FAIL_TOPIC"):
        raise SystemExit(42)
    raise SystemExit(0)
if args[:2] == ["-m", "trec_rag.agentic_retrieval_shard_bundle"]:
    command = args[2]
    if command == "pack":
        topic = args[4]
        archive = pathlib.Path(args[5])
        marker = pathlib.Path(args[6])
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(("archive:" + topic).encode())
        marker.write_text("{}")
    raise SystemExit(0)
raise SystemExit(0)
""",
    )
    _write_executable(
        venv / "hf",
        f"#!{real_python}\n"
        + """import pathlib
import shutil
import sys

runtime = pathlib.Path(__import__("os").environ["FAKE_RUNTIME"])
remote = runtime / "remote"
args = sys.argv[1:]
if args[:2] == ["buckets", "list"]:
    prefix = args[2].rstrip("/")
    topic = prefix.split("/")[-1]
    rows = []
    target = remote / topic
    if target.exists():
        rows = [{"path": prefix + "/" + path.name, "type": "file"} for path in sorted(target.iterdir())]
    print(__import__("json").dumps(rows))
    raise SystemExit(0)
if args[:2] == ["buckets", "sync"]:
    source = pathlib.Path(args[2])
    topic = args[3].rstrip("/").split("/")[-1]
    target = remote / topic
    target.mkdir(exist_ok=True)
    for path in source.iterdir():
        destination = target / path.name
        if not destination.exists():
            shutil.copyfile(path, destination)
    raise SystemExit(0)
if args[:2] == ["buckets", "cp"]:
    source = args[2]
    destination = pathlib.Path(args[3])
    topic, name = source.rstrip("/").split("/")[-2:]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(remote / topic / name, destination)
    raise SystemExit(0)
if args[:2] == ["auth", "whoami"] or args[:2] == ["buckets", "info"] or args == ["buckets", "--help"]:
    print("{\\"private\\":true}")
    raise SystemExit(0)
raise SystemExit(0)
""",
    )
    env = os.environ.copy()
    env.update({
        "PATH": f"{runtime / 'bin'}:{env['PATH']}",
        "FAKE_RUNTIME": str(runtime),
        "FAKE_FAIL_TOPIC": fail_topic,
        "AGENTIC_WORK_ROOT_BASE": str(runtime / "workers"),
        "HF_CLI_MODE": "direct",
        "INDEX_URL": "https://index.invalid",
        "PYSERINI_API_TOKEN": "secret-index",
        "OPENROUTER_API_KEY": "secret-openrouter",
        "HF_TOKEN": "secret-hf",
    })
    return checkout, plan, env


def test_wrapper_runs_topics_sequentially_and_publishes_marker_last(tmp_path: Path) -> None:
    checkout, plan, env = _fake_agentic_checkout(tmp_path)
    result = _run(
        checkout / "code/tools/run_agentic_retrieval_worker.sh",
        "--task-name", "task-0", "--run-id", "run-0", "--plan", str(plan),
        "--plan-sha256", "a" * 64, "--config", "configs/agentic.yaml",
        "--artifact-prefix", "hf://buckets/private/trec_rag_2026/experiments/run-0",
        "--topic", "rag2026-0", "--topic", "rag2026-1", cwd=checkout, env=env,
    )
    assert result.returncode == 0, result.stderr
    rows = (tmp_path / "runtime/events/worker.log").read_text().splitlines()
    assert rows == ["worker|rag2026-0", "worker|rag2026-1"]
    assert "OPENROUTER_API_KEY" not in result.stdout + result.stderr
    assert (tmp_path / "runtime/remote/rag2026-0/bundle-complete.json").exists()
    assert (tmp_path / "runtime/remote/rag2026-1/bundle-complete.json").exists()


def test_wrapper_failure_preserves_prior_topic_and_writes_safe_receipt(tmp_path: Path) -> None:
    checkout, plan, env = _fake_agentic_checkout(tmp_path, fail_topic="rag2026-1")
    result = _run(
        checkout / "code/tools/run_agentic_retrieval_worker.sh",
        "--task-name", "task-0", "--run-id", "run-0", "--plan", str(plan),
        "--plan-sha256", "a" * 64, "--config", "configs/agentic.yaml",
        "--artifact-prefix", "hf://buckets/private/trec_rag_2026/experiments/run-0",
        "--topic", "rag2026-0", "--topic", "rag2026-1", cwd=checkout, env=env,
    )
    assert result.returncode != 0
    assert (tmp_path / "runtime/remote/rag2026-0/bundle-complete.json").exists()
    assert not (tmp_path / "runtime/remote/rag2026-1/bundle-complete.json").exists()
    receipts = list((checkout / "outputs").rglob("failure-receipt.json"))
    assert receipts
    payload = json.loads(receipts[0].read_text())
    assert payload["status"] == "failed"
    assert "secret" not in receipts[0].read_text()
