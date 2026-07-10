#!/usr/bin/env bash
set -euo pipefail

# Auto-select the practical local environment:
# - AMD ROCm host: sync the ROCm reranker environment.
# - Other host: sync the normal project environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

has_amd_rocm_device() {
  local info
  if command -v rocminfo >/dev/null 2>&1; then
    info="$(rocminfo 2>/dev/null || true)"
    if grep -Eiq "gfx11|gfx12|AMD Radeon|AMD Ryzen AI" <<<"${info}"; then
      return 0
    fi
  fi
  if command -v rocm-smi >/dev/null 2>&1; then
    info="$(rocm-smi --showproductname 2>/dev/null || true)"
    if grep -Eiq "AMD|Radeon|Ryzen|gfx11|gfx12" <<<"${info}"; then
      return 0
    fi
  fi
  return 1
}

if has_amd_rocm_device; then
  exec "${SCRIPT_DIR}/setup_rocm_ryzen_env.sh"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
  exit 1
fi

echo "No AMD ROCm device detected; syncing the standard project environment."
uv sync
