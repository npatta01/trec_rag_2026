#!/usr/bin/env bash
set -euo pipefail

# Prove each mutually exclusive hardware group selects the torch build it should.
#
# Read-only by construction: `uv export --frozen` resolves from the committed
# uv.lock and never installs, downloads, mutates the environment, or touches any
# model or retrieval cache. Safe to run beside an active pipeline task.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
  exit 1
fi

ROCM_SOURCE="repo.radeon.com"
status=0

check_group() {
  local group="$1" expectation="$2"
  local export_output requirement

  if ! export_output="$(uv export --frozen --group "${group}" --no-hashes 2>&1)"; then
    echo "FAIL ${group}: uv.lock does not carry this group." >&2
    echo "  Regenerate the lock with 'uv lock' on a host that can reach" >&2
    echo "  ${ROCM_SOURCE}, then re-run this check." >&2
    status=1
    return
  fi

  requirement="$(grep -E '^torch( |@|==)' <<<"${export_output}" || true)"
  if [[ -z "${requirement}" ]]; then
    echo "FAIL ${group}: uv.lock resolves no torch distribution for this group" >&2
    status=1
    return
  fi

  case "${expectation}" in
    rocm)
      if [[ "${requirement}" != *"${ROCM_SOURCE}"* ]]; then
        echo "FAIL ${group}: expected a ${ROCM_SOURCE} wheel, got:" >&2
        echo "  ${requirement}" >&2
        status=1
        return
      fi
      ;;
    cuda)
      if [[ "${requirement}" == *"${ROCM_SOURCE}"* ]]; then
        echo "FAIL ${group}: expected the PyPI CUDA build, got the ROCm wheel:" >&2
        echo "  ${requirement}" >&2
        status=1
        return
      fi
      ;;
    *)
      echo "unknown expectation: ${expectation}" >&2
      exit 1
      ;;
  esac

  echo "OK   ${group}: ${requirement%% ;*}"
}

check_group rocm rocm
check_group cuda cuda

if [[ "${status}" -ne 0 ]]; then
  echo "torch group verification failed" >&2
  exit 1
fi

echo "Both hardware groups select their intended torch source."
