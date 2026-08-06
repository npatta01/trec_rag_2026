#!/usr/bin/env bash
set -euo pipefail

# Prove that uv.lock is current with pyproject.toml and that each mutually
# exclusive hardware group selects the torch build it should.
#
# Nothing here installs: `uv lock --check` only compares a fresh resolution
# against the committed lock, and `uv export --frozen` (both the requirements
# and the PEP 751 `pylock.toml` rendering) reads the lock alone.
# Every uv call runs with `--no-cache` and a throwaway `--cache-dir`, so the
# shared persistent uv cache is neither read nor written, and no model,
# retrieval, or reranker cache is touched. Every call also runs with
# `--no-python-downloads`: this project pins Python 3.12.13, and on a host
# without that interpreter uv would otherwise fetch and install a managed one
# just to resolve, which is exactly the "installs nothing" claim being made
# here. Safe to run beside an active pipeline task. `uv lock --check` does
# re-resolve, so it needs package-index access; `uv export --frozen` does not.

# Located with shell builtins only (`cd`, `pwd`, parameter expansion) rather
# than `dirname`, so that the "uv is required" diagnostic below is the first
# thing that can fail. Calling out to coreutils here would make an empty or
# broken PATH die with `dirname: not found` instead of the message that tells
# the reader what to install.
SCRIPT_PATH="${BASH_SOURCE[0]}"
[[ "${SCRIPT_PATH}" == */* ]] || SCRIPT_PATH="./${SCRIPT_PATH}"
SCRIPT_PARENT="${SCRIPT_PATH%/*}"
SCRIPT_DIR="$(cd "${SCRIPT_PARENT:-/}" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
  exit 1
fi

ROCM_SOURCE="repo.radeon.com"
# The cuda group must resolve this exact PyPI pin: not a URL, not another
# version, not a same-version wheel from some other index.
CUDA_TORCH_VERSION="2.9.1"
CUDA_TORCH_REQUIREMENT="torch==${CUDA_TORCH_VERSION}"
# Where that pin has to come from. A requirements export renders *any* registry
# package as `torch==2.9.1`, whichever index served it, so the requirement line
# alone cannot prove the source. The PEP 751 export can: it records the
# resolved index per package, scoped to the group we asked for.
CUDA_TORCH_INDEX="https://pypi.org/simple"
# A GPU-enabled PyPI torch wheel pulls the NVIDIA runtime in as dependencies.
# A CPU build published under the same version pulls none of them, so their
# presence is what separates "torch==2.9.1" from "torch==2.9.1, but CPU-only".
CUDA_RUNTIME_PACKAGES=(nvidia-cuda-runtime nvidia-cublas nvidia-cudnn)
# uv's own wording when the lock cannot serve the request, as opposed to any
# other failure (network, bad flag, unreadable pyproject). Anything that does
# not match this is reported as-is instead of being blamed on the group.
LOCK_PROBLEM_PATTERN='needs to be updated|not up-to-date|out of date|is not defined|no such group|not found in the lockfile|does not contain'

UV_TMP_CACHE="$(mktemp -d "${TMPDIR:-/tmp}/verify-torch-groups-uv-cache.XXXXXX")"
trap 'rm -rf "${UV_TMP_CACHE}"' EXIT

# Read-only by construction, deliberately cache-less, and forbidden from
# provisioning an interpreter: --no-cache keeps uv from populating a persistent
# cache, --cache-dir keeps whatever it does write inside the temporary directory
# removed on exit, and --no-python-downloads makes uv fail loudly on a host
# missing the pinned Python instead of silently downloading and installing one.
uv_readonly() {
  uv --no-cache --cache-dir "${UV_TMP_CACHE}" --no-python-downloads "$@"
}

emit_uv_output() {
  local line
  echo "  uv reported:" >&2
  if [[ -z "${1//[[:space:]]/}" ]]; then
    echo "    (no output)" >&2
    return
  fi
  while IFS= read -r line; do
    printf '    %s\n' "${line}" >&2
  done <<<"$1"
}

regenerate_hint() {
  echo "  Regenerate the lock with 'uv lock' on a host that can reach" >&2
  echo "  ${ROCM_SOURCE}, then re-run this check." >&2
}

status=0

check_lock_is_current() {
  local check_output
  if check_output="$(uv_readonly lock --check 2>&1)"; then
    echo "OK   uv.lock is current with pyproject.toml"
    return 0
  fi

  # The hint belongs only to a diagnosed stale lock. An unknown flag, a failed
  # authentication, or a dead network says nothing about the lock's currency,
  # and telling the reader to regenerate it would send them to re-resolve
  # against the very index they cannot reach.
  if [[ "${check_output}" =~ ${LOCK_PROBLEM_PATTERN} ]]; then
    echo "FAIL uv.lock is stale: it no longer matches pyproject.toml." >&2
    emit_uv_output "${check_output}"
    regenerate_hint
  else
    echo "FAIL uv could not verify uv.lock against pyproject.toml." >&2
    emit_uv_output "${check_output}"
  fi
  return 1
}

# Print "version<TAB>index" for every top-level [[packages]] entry in a PEP 751
# pylock.toml whose name is "$1" (index empty when the entry has none, i.e. when
# it came from a direct URL or another non-registry source).
#
# Only column-0 keys inside a [[packages]] table are read. Sub-tables such as
# [[packages.wheels]] carry their own `name` and `url`, and an inline
# `wheels = [{ name = "torch-2.9.1-...whl", ... }]` array repeats the wheel
# filename; neither may be mistaken for the package's own fields.
pylock_package_fields() {
  awk -v want="$1" '
    function value(line) {
      sub(/^[^=]*=[ \t]*/, "", line)
      sub(/[ \t]*$/, "", line)
      sub(/^"/, "", line)
      sub(/"$/, "", line)
      return line
    }
    function flush() {
      if (in_package && name == want) { print version "\t" index_url }
      name = ""; version = ""; index_url = ""
    }
    /^\[\[packages\]\][ \t]*$/ { flush(); in_package = 1; in_subtable = 0; next }
    /^\[/ {
      if ($0 ~ /^\[\[?packages\./) { in_subtable = 1 }
      else { flush(); in_package = 0; in_subtable = 0 }
      next
    }
    in_package && !in_subtable && /^name[ \t]*=/ { name = value($0); next }
    in_package && !in_subtable && /^version[ \t]*=/ { version = value($0); next }
    in_package && !in_subtable && /^index[ \t]*=/ { index_url = value($0); next }
    END { flush() }
  '
}

# Prove where the cuda group's torch actually comes from. `uv export` in the
# default requirements format renders every registry package as `name==version`
# with no index attached, so `torch==2.9.1` looks identical whether PyPI or some
# mirror served it. The PEP 751 export is source-bearing, and asking for it with
# `--group cuda` scopes it to this group's resolution instead of leaving us to
# guess which fork of an unscoped multi-fork uv.lock block applies.
check_cuda_torch_source() {
  local group="$1"
  local pylock_output fields entry_count version index_url

  if ! pylock_output="$(uv_readonly export --frozen --group "${group}" --format pylock.toml 2>&1)"; then
    if [[ "${pylock_output}" =~ ${LOCK_PROBLEM_PATTERN} ]]; then
      echo "FAIL ${group}: uv.lock does not carry this group." >&2
      emit_uv_output "${pylock_output}"
      regenerate_hint
    else
      echo "FAIL ${group}: uv export --format pylock.toml failed for a reason unrelated to this group." >&2
      emit_uv_output "${pylock_output}"
    fi
    return 1
  fi

  fields="$(pylock_package_fields torch <<<"${pylock_output}")"
  if [[ -z "${fields}" ]]; then
    echo "FAIL ${group}: the pylock.toml export names no torch package, so its source" >&2
    echo "  cannot be proved." >&2
    emit_uv_output "${pylock_output}"
    return 1
  fi
  entry_count="$(grep -c '' <<<"${fields}")"
  if [[ "${entry_count}" -ne 1 ]]; then
    echo "FAIL ${group}: expected one torch package in the pylock.toml export, got ${entry_count}:" >&2
    while IFS=$'\t' read -r version index_url; do
      echo "    version ${version:-(none)}, index ${index_url:-(none)}" >&2
    done <<<"${fields}"
    return 1
  fi

  version="${fields%%$'\t'*}"
  index_url="${fields#*$'\t'}"

  if [[ "${version}" != "${CUDA_TORCH_VERSION}" ]]; then
    echo "FAIL ${group}: the pylock.toml export selects torch ${version:-(no version)}, not ${CUDA_TORCH_VERSION}." >&2
    return 1
  fi
  if [[ -z "${index_url}" ]]; then
    echo "FAIL ${group}: the pylock.toml export records no index for torch ${version}, so it comes" >&2
    echo "  from a direct URL or another non-registry source rather than ${CUDA_TORCH_INDEX}." >&2
    return 1
  fi
  if [[ "${index_url}" != "${CUDA_TORCH_INDEX}" ]]; then
    echo "FAIL ${group}: torch ${version} resolves from ${index_url}, not ${CUDA_TORCH_INDEX}." >&2
    return 1
  fi
  return 0
}

check_group() {
  local group="$1" expectation="$2"
  local export_output torch_lines requirement package detail=""
  local missing=()

  if ! export_output="$(uv_readonly export --frozen --group "${group}" --no-hashes 2>&1)"; then
    if [[ "${export_output}" =~ ${LOCK_PROBLEM_PATTERN} ]]; then
      echo "FAIL ${group}: uv.lock does not carry this group." >&2
      emit_uv_output "${export_output}"
      regenerate_hint
    else
      echo "FAIL ${group}: uv export failed for a reason unrelated to this group." >&2
      emit_uv_output "${export_output}"
    fi
    status=1
    return
  fi

  torch_lines="$(grep -E '^torch([[:space:]]|@|==)' <<<"${export_output}" || true)"
  if [[ -z "${torch_lines}" ]]; then
    echo "FAIL ${group}: uv.lock resolves no torch distribution for this group" >&2
    status=1
    return
  fi
  if [[ "$(grep -c '' <<<"${torch_lines}")" -ne 1 ]]; then
    echo "FAIL ${group}: expected one torch requirement, got:" >&2
    emit_uv_output "${torch_lines}"
    status=1
    return
  fi

  # Drop the environment marker and any trailing whitespace.
  requirement="${torch_lines%%;*}"
  requirement="${requirement%"${requirement##*[![:space:]]}"}"

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
        echo "FAIL ${group}: expected ${CUDA_TORCH_REQUIREMENT} from PyPI, got the ROCm wheel:" >&2
        echo "  ${requirement}" >&2
        status=1
        return
      fi
      if [[ "${requirement}" == *"@"* ]]; then
        if [[ "${requirement}" == *"+cpu"* || "${requirement}" == *"%2Bcpu"* || "${requirement}" == *"/cpu/"* ]]; then
          echo "FAIL ${group}: expected ${CUDA_TORCH_REQUIREMENT} from PyPI, got a CPU-only wheel:" >&2
        else
          echo "FAIL ${group}: expected the PyPI pin ${CUDA_TORCH_REQUIREMENT}, got a direct URL:" >&2
        fi
        echo "  ${requirement}" >&2
        status=1
        return
      fi
      if [[ "${requirement}" != "${CUDA_TORCH_REQUIREMENT}" ]]; then
        echo "FAIL ${group}: expected exactly ${CUDA_TORCH_REQUIREMENT}, got:" >&2
        echo "  ${requirement}" >&2
        status=1
        return
      fi
      for package in "${CUDA_RUNTIME_PACKAGES[@]}"; do
        if ! grep -qE "^${package}(-cu[0-9]+)?([[:space:]]|@|==)" <<<"${export_output}"; then
          missing+=("${package}")
        fi
      done
      if [[ "${#missing[@]}" -gt 0 ]]; then
        echo "FAIL ${group}: ${requirement} resolves without the NVIDIA runtime packages a" >&2
        echo "  GPU-enabled wheel depends on. Missing: ${missing[*]}." >&2
        echo "  That is the signature of a CPU-only build published under the same version." >&2
        status=1
        return
      fi
      # Everything above reads the requirement line, which cannot name a source.
      if ! check_cuda_torch_source "${group}"; then
        status=1
        return
      fi
      detail=" from ${CUDA_TORCH_INDEX}"
      ;;
    *)
      echo "unknown expectation: ${expectation}" >&2
      exit 1
      ;;
  esac

  echo "OK   ${group}: ${requirement}${detail}"
}

# A stale lock makes every export below meaningless, so stop at the first failure.
if ! check_lock_is_current; then
  echo "torch group verification failed" >&2
  exit 1
fi

check_group rocm rocm
check_group cuda cuda

if [[ "${status}" -ne 0 ]]; then
  echo "torch group verification failed" >&2
  exit 1
fi

echo "Both hardware groups select their intended torch source."
