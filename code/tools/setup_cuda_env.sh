#!/usr/bin/env bash
set -euo pipefail

# Reproducible CUDA/PyTorch env for rented NVIDIA boxes that run a subset of the
# competition topics before the artifact is downloaded back to the ROCm host.
# The pinned versions come from the `cuda` dependency group, which mirrors what
# the `rocm` group resolves to, so both hosts write interchangeable reranker
# score-cache entries.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
  exit 1
fi

# --locked keeps this host from re-resolving the ROCm branch of the lock, whose
# repo.radeon.com wheels are usually unreachable from a rented CUDA box.
if ! uv sync --group cuda --locked; then
  echo "uv.lock does not match pyproject.toml." >&2
  echo "Regenerate it with 'uv lock' on the ROCm host and commit the result;" >&2
  echo "locking here would need repo.radeon.com for the rocm group." >&2
  exit 1
fi

VENV_ABS="${REPO_ROOT}/.venv"

# competition_retrieval loads the reranker with local_files_only=True at a
# pinned revision, so the weights must already be in the HF cache.
"${VENV_ABS}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download

from trec_rag.mixedbread_passage_scorer import MIXEDBREAD_MODEL, MIXEDBREAD_REVISION

path = snapshot_download(MIXEDBREAD_MODEL, revision=MIXEDBREAD_REVISION)
print(f"reranker_weights={path}")
PY

"${VENV_ABS}/bin/python" - <<'PY'
import sys

import torch

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("CUDA is unavailable; passage.device: cuda runs will fail here.", file=sys.stderr)
    raise SystemExit(2)
print(f"torch.cuda.device_name={torch.cuda.get_device_name(0)}")
PY

echo "CUDA environment ready: ${VENV_ABS}/bin/python"
