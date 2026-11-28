"""Own a local vLLM lifecycle and run the config-pinned pointwise scorer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import requests


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from trec_rag.retrieval_ranking_benchmark import load_benchmark_config  # noqa: E402


def _command(settings: dict[str, object]) -> list[str]:
    server = settings["local_vllm"]["server"]
    vllm = Path(sys.executable).with_name("vllm")
    command = [
        str(vllm),
        "serve",
        str(settings["model"]),
        "--served-model-name",
        str(settings["model"]),
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--dtype",
        str(settings["dtype"]),
        "--max-model-len",
        str(settings["max_length"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--generation-config",
        "vllm",
        "--offload-backend",
        str(server["offload_backend"]),
        "--offload-group-size",
        str(server["offload_group_size"]),
        "--offload-num-in-group",
        str(server["offload_num_in_group"]),
        "--offload-prefetch-step",
        str(server["offload_prefetch_step"]),
    ]
    if server["enforce_eager"]:
        command.append("--enforce-eager")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("smoke", "full"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    settings = dict(config.raw["organizer_pointwise"])
    local = settings["local_vllm"]
    server = local["server"]
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in server["environment"].items()})
    environment["PYTHONPATH"] = str(config.repo_root / "code")
    log_path = config.path("output_dir") / f"local_vllm_{args.operation}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    scorer = Path(__file__).with_name("run_qwen_pointwise_vllm.py")
    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                _command(settings),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            deadline = time.monotonic() + float(server["startup_timeout_seconds"])
            health = f"http://127.0.0.1:{server['port']}/health"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited during startup; inspect {log_path}")
                try:
                    response = requests.get(health, timeout=2)
                    if response.ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(f"vLLM did not become healthy; inspect {log_path}")
            completed = subprocess.run(
                [sys.executable, str(scorer), args.operation, "--config", str(config.source_path)],
                check=False,
                env=environment,
            )
            if completed.returncode:
                raise RuntimeError(f"pointwise scorer exited {completed.returncode}")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    receipt_name = (
        "local_pointwise_smoke.json"
        if args.operation == "smoke"
        else "local_pointwise_runtime.json"
    )
    receipt_path = config.path("output_dir") / receipt_name
    print(json.dumps(json.loads(receipt_path.read_text(encoding="utf-8")), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
