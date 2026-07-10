"""Small ROCm/PyTorch health check for Ryzen APU reranking environments."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def _first_existing(paths: list[str]) -> str:
    for path in paths:
        if Path(path).exists():
            return path
    return ""


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"kernel={_run(['uname', '-r'])}")
    print(f"rocm_smi={_run(['rocm-smi', '--showdriverversion'])}")
    amdgpu_ids = _first_existing(
        [
            "/opt/amdgpu/share/libdrm/amdgpu.ids",
            "/opt/rocm/core-7.13/lib/rocm_sysdeps/share/libdrm/amdgpu.ids",
            "/usr/share/libdrm/amdgpu.ids",
        ]
    )
    print(f"amdgpu_ids={amdgpu_ids or 'not found'}")

    try:
        import torch
    except ImportError as exc:
        print(f"torch_import_error={exc}")
        return 1

    print(f"torch={torch.__version__}")
    print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        return 2

    print(f"torch.cuda.device_name={torch.cuda.get_device_name(0)}")
    try:
        x = torch.ones((128, 128), device="cuda", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"gpu_tensor_error={type(exc).__name__}: {exc}")
        return 3
    print(f"gpu_tensor_ok={float(y[0, 0].item())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
