#!/bin/bash
# Performance regression test: compare current checkout vs origin/main.
#
# This script orchestrates the newer Python primitives:
#   - run_current_regression.py: run/cache one checkout's regression results
#   - compare_perf_regression.py: compare two JSON result payloads
#
# Usage:
#   ./maint/scripts/run_perf_regression.sh [run_current_regression filters...]
#
# Examples:
#   ./maint/scripts/run_perf_regression.sh -k gemm/regression_example_gemm.py
#   ./maint/scripts/run_perf_regression.sh -k flash_attention/regression_example_flash_attention.py::example_mha_fwd_bshd
#
# Environment variables:
#   PYTHON_VERSION       Python version for venvs (default: 3.12)
#   WORK_DIR             Working directory (default: ${TMPDIR:-/tmp}/tilelang-perf-regression-$USER/<repo>-<hash>)
#   BASE_REF             Baseline git ref (default: origin/main)
#   SKIP_BUILD_NEW       Reuse current venv when set to 1
#   SKIP_BUILD_OLD       Reuse baseline venv when set to 1
#   REFRESH              Rerun selected cases and overwrite perf cache when set to 1
#   NO_CACHE             Disable perf result cache when set to 1
#   FAIL_ON_ERROR        Exit non-zero if either run has failed cases when set to 1
#   FAIL_ON_REGRESSION   Exit non-zero if any common result regresses when set to 1
#   REGRESSION_THRESHOLD Speedup threshold for FAIL_ON_REGRESSION (default: 1.0)
#   FAIL_ON_MISSING      Exit non-zero if one side is missing results when set to 1
#   CMAKE_GENERATOR      CMake generator for package builds (default: Ninja)

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,35p' "$0"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TMP_ROOT="${TMPDIR:-/tmp}"
RUN_USER="${USER:-$(id -un 2>/dev/null || echo user)}"
REPO_NAME="$(basename "${REPO_ROOT}")"
if command -v sha256sum >/dev/null 2>&1; then
    REPO_HASH="$(printf '%s' "${REPO_ROOT}" | sha256sum | cut -c 1-12)"
else
    REPO_HASH="$(printf '%s' "${REPO_ROOT}" | cksum | awk '{print $1}')"
fi
DEFAULT_WORK_DIR="${TMP_ROOT%/}/tilelang-perf-regression-${RUN_USER}/${REPO_NAME}-${REPO_HASH}"
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"
BASE_REF="${BASE_REF:-origin/main}"
BASE_LABEL="${BASE_LABEL:-${BASE_REF}}"
CURRENT_LABEL="${CURRENT_LABEL:-current}"

SKIP_BUILD_NEW="${SKIP_BUILD_NEW:-0}"
SKIP_BUILD_OLD="${SKIP_BUILD_OLD:-0}"
REFRESH="${REFRESH:-0}"
NO_CACHE="${NO_CACHE:-0}"
FAIL_ON_ERROR="${FAIL_ON_ERROR:-0}"
FAIL_ON_REGRESSION="${FAIL_ON_REGRESSION:-0}"
REGRESSION_THRESHOLD="${REGRESSION_THRESHOLD:-1.0}"
FAIL_ON_MISSING="${FAIL_ON_MISSING:-0}"
UPDATE_SUBMODULES="${UPDATE_SUBMODULES:-1}"

mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"

CURRENT_VENV="${WORK_DIR}/venvs/current"
BASE_VENV="${WORK_DIR}/venvs/baseline"
CURRENT_BUILD_DIR="${WORK_DIR}/builds/current"
BASE_BUILD_DIR="${WORK_DIR}/builds/baseline"
BASE_WORKTREE="${WORK_DIR}/worktrees/baseline"

CURRENT_JSON="${WORK_DIR}/current_result.json"
BASE_JSON="${WORK_DIR}/baseline_result.json"
RESULT_MD="${WORK_DIR}/regression_result.md"
RESULT_PNG="${WORK_DIR}/regression_result.png"
RESULT_JSON="${WORK_DIR}/regression_result.json"

REGRESSION_ARGS=("$@")

handle_interrupt() {
    local status=$?
    echo ""
    echo "Interrupted."
    echo "No checkout/stash cleanup is needed; the current worktree was not switched."
    echo "Partial state is kept under: ${WORK_DIR}"
    echo "A later run will recreate the baseline worktree and, unless SKIP_BUILD_* is set, rebuild venvs."
    echo "To clean everything manually: rm -rf ${WORK_DIR}"
    exit "${status}"
}

trap handle_interrupt INT TERM

echo "============================================"
echo "TileLang Performance Regression"
echo "============================================"
echo "Repo root:      ${REPO_ROOT}"
echo "Work dir:       ${WORK_DIR}"
echo "Python version: ${PYTHON_VERSION}"
echo "Baseline ref:   ${BASE_REF}"
echo "Filters:        ${REGRESSION_ARGS[*]:-(all)}"
echo ""

cd "${REPO_ROOT}"

fetch_baseline() {
    if [[ "${BASE_REF}" == */* ]]; then
        local remote="${BASE_REF%%/*}"
        local branch="${BASE_REF#*/}"
        if git remote get-url "${remote}" >/dev/null 2>&1; then
            echo "Fetching ${remote}/${branch}..."
            git fetch "${remote}" "${branch}"
        fi
    fi
    git rev-parse --verify "${BASE_REF}^{commit}" >/dev/null
}

prepare_baseline_worktree() {
    echo ""
    echo "============================================"
    echo "Preparing baseline worktree (${BASE_REF})"
    echo "============================================"

    if git worktree list --porcelain | grep -Fx "worktree ${BASE_WORKTREE}" >/dev/null 2>&1; then
        git worktree remove --force "${BASE_WORKTREE}"
    else
        rm -rf "${BASE_WORKTREE}"
    fi

    mkdir -p "$(dirname "${BASE_WORKTREE}")"
    git worktree add --detach "${BASE_WORKTREE}" "${BASE_REF}"

    if [[ "${UPDATE_SUBMODULES}" == "1" ]]; then
        git -C "${BASE_WORKTREE}" submodule update --init --recursive
    fi
}

create_venv() {
    local venv_dir="$1"
    rm -rf "${venv_dir}"
    mkdir -p "$(dirname "${venv_dir}")"

    if command -v uv >/dev/null 2>&1; then
        uv venv --python "${PYTHON_VERSION}" "${venv_dir}"
    else
        if command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
            "python${PYTHON_VERSION}" -m venv "${venv_dir}"
        else
            python -m venv "${venv_dir}"
        fi
    fi
}

install_repo() {
    local label="$1"
    local repo_dir="$2"
    local venv_dir="$3"
    local build_dir="$4"
    local skip_build="$5"

    echo ""
    echo "============================================"
    echo "Building ${label}"
    echo "============================================"
    echo "Repo: ${repo_dir}"
    echo "Venv: ${venv_dir}"
    echo "Build dir: ${build_dir}"

    if [[ "${skip_build}" == "1" ]]; then
        if [[ ! -x "${venv_dir}/bin/python" ]]; then
            echo "Requested skip build for ${label}, but ${venv_dir}/bin/python does not exist." >&2
            exit 1
        fi
        echo "Skipping build for ${label}; reusing existing venv."
        return
    fi

    create_venv "${venv_dir}"
    rm -rf "${build_dir}"
    mkdir -p "$(dirname "${build_dir}")"
    (
        cd "${repo_dir}"
        source "${venv_dir}/bin/activate"
        export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
        unset CMAKE_MAKE_PROGRAM
        if command -v uv >/dev/null 2>&1; then
            uv pip install -v -r requirements-test.txt
            uv pip install -v -C "build-dir=${build_dir}" .
        else
            python -m pip install -v -r requirements-test.txt
            python -m pip install -v -C "build-dir=${build_dir}" .
        fi
    )
}

run_regression() {
    local label="$1"
    local repo_dir="$2"
    local python_bin="$3"
    local output_json="$4"

    local runner_flags=("--output" "${output_json}" "--format" "markdown")
    if [[ "${REFRESH}" == "1" ]]; then
        runner_flags+=("--refresh")
    fi
    if [[ "${NO_CACHE}" == "1" ]]; then
        runner_flags+=("--no-cache")
    fi

    echo ""
    echo "============================================"
    echo "Running ${label} regression"
    echo "============================================"
    (
        cd "${repo_dir}"
        "${python_bin}" "${repo_dir}/maint/scripts/run_current_regression.py" \
            "${runner_flags[@]}" \
            "${REGRESSION_ARGS[@]}"
    )
}

compare_results() {
    local compare_flags=(
        "${BASE_JSON}"
        "${CURRENT_JSON}"
        "--old-label" "${BASE_LABEL}"
        "--new-label" "${CURRENT_LABEL}"
        "--output-md" "${RESULT_MD}"
        "--output-json" "${RESULT_JSON}"
        "--output-png" "${RESULT_PNG}"
    )

    if [[ "${FAIL_ON_ERROR}" == "1" ]]; then
        compare_flags+=("--fail-on-failure")
    fi
    if [[ "${FAIL_ON_REGRESSION}" == "1" ]]; then
        compare_flags+=("--fail-on-regression" "--regression-threshold" "${REGRESSION_THRESHOLD}")
    fi
    if [[ "${FAIL_ON_MISSING}" == "1" ]]; then
        compare_flags+=("--fail-on-missing")
    fi

    echo ""
    echo "============================================"
    echo "Comparing results"
    echo "============================================"
    "${CURRENT_VENV}/bin/python" "${SCRIPT_DIR}/compare_perf_regression.py" "${compare_flags[@]}"
}

fetch_baseline
prepare_baseline_worktree

if [[ "${UPDATE_SUBMODULES}" == "1" ]]; then
    git -C "${REPO_ROOT}" submodule update --init --recursive
fi

install_repo "${CURRENT_LABEL}" "${REPO_ROOT}" "${CURRENT_VENV}" "${CURRENT_BUILD_DIR}" "${SKIP_BUILD_NEW}"
install_repo "${BASE_LABEL}" "${BASE_WORKTREE}" "${BASE_VENV}" "${BASE_BUILD_DIR}" "${SKIP_BUILD_OLD}"

run_regression "${CURRENT_LABEL}" "${REPO_ROOT}" "${CURRENT_VENV}/bin/python" "${CURRENT_JSON}"
run_regression "${BASE_LABEL}" "${BASE_WORKTREE}" "${BASE_VENV}/bin/python" "${BASE_JSON}"

compare_results

echo ""
echo "============================================"
echo "Results"
echo "============================================"
echo "Current JSON:  ${CURRENT_JSON}"
echo "Baseline JSON: ${BASE_JSON}"
echo "Markdown:      ${RESULT_MD}"
echo "Plot:          ${RESULT_PNG}"
echo "Compare JSON:  ${RESULT_JSON}"
echo ""
cat "${RESULT_MD}"
