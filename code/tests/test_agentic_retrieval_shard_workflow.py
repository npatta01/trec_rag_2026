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


def test_agentic_worker_uses_exact_repo_python_version() -> None:
    script = WRAPPER_PATH.read_text(encoding="utf-8")
    required = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()

    assert required == "3.12.13"
    assert 'required_python=$(<"$REPO_ROOT/.python-version")' in script
    assert '--python "$required_python"' in script
    assert "--no-python-downloads" not in script
    assert "python_runtime_verified=$required_python" in script


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
    shutil.copy2(REPO_ROOT / ".python-version", checkout / ".python-version")
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
    fake_bin = runtime / "bin"
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
if args and args[0] == "-c":
    print("3.12.13")
    raise SystemExit(0)
if args and args[0] == "-":
    model, revision = args[1:3]
    events = runtime / "events"
    events.mkdir(exist_ok=True)
    with (events / "worker.log").open("a") as stream:
        stream.write("prefetch|" + model + "|" + revision + "\\n")
    print("model_snapshot_verified=" + model + "@" + revision)
    raise SystemExit(0)
if args[:2] == ["-m", "trec_rag.competition_agentic_worker"]:
    topic = args[args.index("--topic") + 1]
    events = runtime / "events"
    events.mkdir(exist_ok=True)
    if pathlib.Path("outputs/run-0/work").exists():
        with (events / "worker.log").open("a") as stream:
            stream.write("output-preexisting\\n")
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
    _write_executable(
        fake_bin / "python3",
        f"""#!/usr/bin/env bash
if [[ ${{1:-}} == -c ]]; then
  printf '3.12\\n'
else
  exec {REPO_ROOT / '.venv/bin/python'} \"$@\"
fi
""",
    )
    _write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'uv-sync\\n' >>"$FAKE_RUNTIME/events.log"
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
    assert rows == [
        "prefetch|mixedbread-ai/mxbai-rerank-base-v2|3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
        "worker|rag2026-0",
        "worker|rag2026-1",
    ]
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
    receipts = list((tmp_path / "runtime/workers").rglob("failure-receipt.json"))
    assert receipts
    payload = json.loads(receipts[0].read_text())
    assert payload["status"] == "failed"
    assert "secret" not in receipts[0].read_text()


def test_wrapper_keeps_output_namespace_empty_until_worker_installs_plan(tmp_path: Path) -> None:
    checkout, plan, env = _fake_agentic_checkout(tmp_path, fail_topic="rag2026-0")
    result = _run(
        checkout / "code/tools/run_agentic_retrieval_worker.sh",
        "--task-name", "task-0", "--run-id", "run-0", "--plan", str(plan),
        "--plan-sha256", "a" * 64, "--config", "configs/agentic.yaml",
        "--artifact-prefix", "hf://buckets/private/trec_rag_2026/experiments/run-0",
        "--topic", "rag2026-0", cwd=checkout, env=env,
    )
    assert result.returncode != 0
    events = (tmp_path / "runtime/events/worker.log").read_text(encoding="utf-8").splitlines()
    assert "output-preexisting" not in events
    receipt = tmp_path / "runtime/workers/run-0/task-0/failures/rag2026-0/task-0/failure-receipt.json"
    assert receipt.is_file()
    assert (tmp_path / "runtime/remote/task-0/failure-receipt.json").is_file()


def test_launcher_runs_dstack_from_transport_directory_for_preview(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    for relative in (
        Path("code/tools/apply_agentic_retrieval_worker.sh"),
        Path("code/tools/run_agentic_retrieval_worker.sh"),
        Path(".dstack/rag26-agentic-retrieval-worker.yaml"),
        Path("configs/agentic.yaml"),
    ):
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = {
            Path("code/tools/apply_agentic_retrieval_worker.sh"): LAUNCHER_PATH,
            Path("code/tools/run_agentic_retrieval_worker.sh"): WRAPPER_PATH,
            Path(".dstack/rag26-agentic-retrieval-worker.yaml"): CONFIG_PATH,
            Path("configs/agentic.yaml"): AGENTIC_CONFIG,
        }[relative]
        shutil.copy2(source, destination)
    (checkout / "plan.json").write_text(
        json.dumps({
            "plan_sha256": "a" * 64,
            "run_id": "run-0",
            "planned_topic_ids": ["rag2026-0"],
        }),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "--quiet", "--initial-branch", "task"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "--quiet", "--no-gpg-sign", "-m", "fixture"], cwd=checkout, check=True)
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(bare)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=checkout, check=True)
    subprocess.run(["git", "push", "--quiet", "--set-upstream", "origin", "task"], cwd=checkout, check=True)
    subprocess.run(["git", "remote", "set-url", "origin", "https://github.com/npatta01/trec_rag_2026.git"], cwd=checkout, check=True)

    (checkout / ".venv/bin").mkdir(parents=True)
    _write_executable(
        checkout / ".venv/bin/python",
        f"#!/usr/bin/env bash\nexec {REPO_ROOT / '.venv/bin/python'} \"$@\"\n",
    )
    _write_executable(checkout / ".venv/bin/hf", "#!/usr/bin/env bash\nexit 0\n")
    (checkout / ".git/info/exclude").write_text("/.venv/\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "dstack",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --version ]]; then
  printf '0.20.29\\n'
  exit 0
fi
[[ ${1:-} == apply && ${2:-} == -f ]]
task_config=$3
python3 - "$task_config" "$@" <<'PY'
from pathlib import Path
import sys
import yaml
config = Path(sys.argv[1]).resolve()
cwd = Path.cwd().resolve()
config.relative_to(cwd)
assert config.parent == cwd
value = yaml.safe_load(config.read_text(encoding="utf-8"))
files = value["files"]
assert {row["path"] for row in files} == {
    "/dstack/run/trec_rag_2026/.agentic-private/run-plan.json",
    "/dstack/run/trec_rag_2026/.agentic-private/agentic-config.yaml",
}
for row in files:
    local = Path(row["local_path"])
    assert local.is_file() and not local.is_symlink()
    assert ".env" not in local.name
argv = sys.argv[2:]
marker = argv.index("--")
run_args = argv[marker + 1:]
assert run_args[run_args.index("--plan") + 1] == ".agentic-private/run-plan.json"
assert run_args[run_args.index("--config") + 1] == ".agentic-private/agentic-config.yaml"
assert not any("trec-rag-agentic-transport" in value for value in run_args)
PY
printf 'fake-preview-ok\\n'
""",
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = _run(
        checkout / "code/tools/apply_agentic_retrieval_worker.sh",
        "--preview", "--name", "agentic-preview-0",
        "--task-name", "task-0", "--run-id", "run-0",
        "--plan-sha256", "a" * 64,
        "--artifact-prefix", "hf://buckets/private/trec_rag_2026/experiments/run-0",
        "--plan", "plan.json", "--config", "configs/agentic.yaml",
        "--topic", "rag2026-0", cwd=checkout, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "fake-preview-ok" in result.stdout


def test_wrapper_bootstraps_locked_environment_before_using_venv(tmp_path: Path) -> None:
    checkout, plan, env = _fake_agentic_checkout(tmp_path)
    runtime = tmp_path / "runtime"
    fake_bin = runtime / "bin"
    prebuilt_python = runtime / "prebuilt-python"
    prebuilt_hf = runtime / "prebuilt-hf"
    shutil.copy2(checkout / ".venv/bin/python", prebuilt_python)
    shutil.copy2(checkout / ".venv/bin/hf", prebuilt_hf)
    shutil.rmtree(checkout / ".venv")
    _write_executable(
        fake_bin / "python3",
        f"""#!/usr/bin/env bash
if [[ ${{1:-}} == -c ]]; then
  printf '3.12\\n'
else
  exec {REPO_ROOT / '.venv/bin/python'} \"$@\"
fi
""",
    )
    _write_executable(
        fake_bin / "uv",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'uv-sync\\n' >"$FAKE_RUNTIME/uv-sync"
mkdir -p .venv/bin
cp {prebuilt_python} .venv/bin/python
cp {prebuilt_hf} .venv/bin/hf
chmod 0755 .venv/bin/python .venv/bin/hf
""",
    )
    result = _run(
        checkout / "code/tools/run_agentic_retrieval_worker.sh",
        "--task-name", "task-0", "--run-id", "run-0", "--plan", str(plan),
        "--plan-sha256", "a" * 64, "--config", "configs/agentic.yaml",
        "--artifact-prefix", "hf://buckets/private/trec_rag_2026/experiments/run-0",
        "--topic", "rag2026-0", cwd=checkout, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert (runtime / "uv-sync").read_text(encoding="utf-8") == "uv-sync\n"
    assert (runtime / "events/worker.log").read_text(encoding="utf-8").splitlines() == [
        "prefetch|mixedbread-ai/mxbai-rerank-base-v2|3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
        "worker|rag2026-0",
    ]


def test_wrapper_prefetches_pinned_mixedbread_before_topic_execution(tmp_path: Path) -> None:
    checkout, plan, env = _fake_agentic_checkout(tmp_path)
    result = _run(
        checkout / "code/tools/run_agentic_retrieval_worker.sh",
        "--task-name", "task-0", "--run-id", "run-0", "--plan", str(plan),
        "--plan-sha256", "a" * 64, "--config", "configs/agentic.yaml",
        "--artifact-prefix", "hf://buckets/private/trec_rag_2026/experiments/run-0",
        "--topic", "rag2026-0", cwd=checkout, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "model_snapshot_verified=mixedbread-ai/mxbai-rerank-base-v2@3ea9d4dffa7d12a4f366be8e275c349de9fc9865" in result.stdout
    rows = (tmp_path / "runtime/events/worker.log").read_text(encoding="utf-8").splitlines()
    assert rows == [
        "prefetch|mixedbread-ai/mxbai-rerank-base-v2|3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
        "worker|rag2026-0",
    ]
