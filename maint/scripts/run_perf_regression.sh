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
#   WHEEL_CACHE          Reuse/build cached TileLang wheels when set to 1 (default: 1)
#   WHEEL_CACHE_DIR      Wheel cache directory (default: $HOME/.tilelang/perf-regression/wheels)
#   REFRESH_WHEEL_CACHE  Rebuild selected wheels even if cached when set to 1
#   REFRESH              Rerun selected cases and overwrite perf cache when set to 1
#   NO_CACHE             Disable perf result cache when set to 1
#   FAIL_ON_ERROR        Exit non-zero if either run has failed cases when set to 1
#   FAIL_ON_REGRESSION   Exit non-zero if any common result regresses when set to 1
#   REGRESSION_THRESHOLD Speedup threshold for FAIL_ON_REGRESSION (default: 1.0)
#   FAIL_ON_MISSING      Exit non-zero if one side is missing results when set to 1
#   CMAKE_GENERATOR      CMake generator for package builds (default: Ninja)
#   INHERIT_PYTHONPATH   Keep caller PYTHONPATH when set to 1 (default: 0)
#   EXTRA_PYTHONPATH     PYTHONPATH to expose to regression/compare processes

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    awk '/^set -euo pipefail$/ {exit} {print}' "$0"
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
WHEEL_CACHE="${WHEEL_CACHE:-1}"
REFRESH_WHEEL_CACHE="${REFRESH_WHEEL_CACHE:-0}"
REFRESH="${REFRESH:-0}"
NO_CACHE="${NO_CACHE:-0}"
FAIL_ON_ERROR="${FAIL_ON_ERROR:-0}"
FAIL_ON_REGRESSION="${FAIL_ON_REGRESSION:-0}"
REGRESSION_THRESHOLD="${REGRESSION_THRESHOLD:-1.0}"
FAIL_ON_MISSING="${FAIL_ON_MISSING:-0}"
UPDATE_SUBMODULES="${UPDATE_SUBMODULES:-1}"
INHERIT_PYTHONPATH="${INHERIT_PYTHONPATH:-0}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"

if [[ -z "${WHEEL_CACHE_DIR:-}" ]]; then
    if [[ -n "${HOME:-}" ]]; then
        WHEEL_CACHE_DIR="${HOME%/}/.tilelang/perf-regression/wheels"
    else
        WHEEL_CACHE_DIR="${WORK_DIR%/}/wheel-cache"
    fi
fi

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

run_isolated_python() {
    if [[ "${INHERIT_PYTHONPATH}" == "1" ]]; then
        if [[ -n "${EXTRA_PYTHONPATH}" ]]; then
            PYTHONPATH="${EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" "$@"
        else
            "$@"
        fi
    elif [[ -n "${EXTRA_PYTHONPATH}" ]]; then
        PYTHONPATH="${EXTRA_PYTHONPATH}" "$@"
    else
        env -u PYTHONPATH "$@"
    fi
}

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
echo "Wheel cache:    ${WHEEL_CACHE} (${WHEEL_CACHE_DIR})"
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

hash_stdin() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

repo_has_untracked_files() {
    local repo_dir="$1"
    if [[ -n "$(git -C "${repo_dir}" ls-files --others --exclude-standard)" ]]; then
        return 0
    fi

    local submodule_status
    submodule_status="$(
        git -C "${repo_dir}" submodule foreach --quiet --recursive \
            'git ls-files --others --exclude-standard | sed "s#^#$name/#"' 2>/dev/null || true
    )"
    [[ -n "${submodule_status}" ]]
}

repo_source_key() {
    local repo_dir="$1"
    local commit
    commit="$(git -C "${repo_dir}" rev-parse --verify HEAD)"

    if git -C "${repo_dir}" diff --quiet --ignore-submodules=none &&
        git -C "${repo_dir}" diff --cached --quiet --ignore-submodules=none &&
        [[ -z "$(git -C "${repo_dir}" status --porcelain=v1 --untracked-files=no)" ]]; then
        printf '%s-clean\n' "${commit}"
        return
    fi

    local digest
    digest="$(
        {
            git -C "${repo_dir}" rev-parse --verify HEAD
            git -C "${repo_dir}" status --porcelain=v1 --untracked-files=no
            git -C "${repo_dir}" diff --binary --no-ext-diff --submodule=short
            git -C "${repo_dir}" diff --cached --binary --no-ext-diff --submodule=short
            git -C "${repo_dir}" submodule status --recursive 2>/dev/null || true
            git -C "${repo_dir}" submodule foreach --quiet --recursive '
                printf "submodule %s\n" "$name"
                git rev-parse --verify HEAD
                git status --porcelain=v1 --untracked-files=no
                git diff --binary --no-ext-diff
                git diff --cached --binary --no-ext-diff
            ' 2>/dev/null || true
        } | hash_stdin
    )"
    printf '%s-dirty-%s\n' "${commit}" "${digest:0:16}"
}

wheel_cache_metadata() {
    local repo_dir="$1"
    local python_bin="$2"

    printf 'schema=tilelang-perf-wheel-cache-v1\n'
    printf 'source_key=%s\n' "$(repo_source_key "${repo_dir}")"
    "${python_bin}" - <<'PY'
import platform
import sys
import sysconfig

print(f"python_version={sys.version.split()[0]}")
print(f"python_cache_tag={sys.implementation.cache_tag or ''}")
print(f"python_soabi={sysconfig.get_config_var('SOABI') or ''}")
print(f"python_platform={sysconfig.get_platform()}")
print(f"platform_system={platform.system()}")
print(f"platform_machine={platform.machine()}")
PY
    printf 'uname_s=%s\n' "$(uname -s)"
    printf 'uname_m=%s\n' "$(uname -m)"
    if command -v cmake >/dev/null 2>&1; then
        cmake --version | sed -n '1s/^cmake version /cmake_version=/p'
    fi
    if command -v ninja >/dev/null 2>&1; then
        printf 'ninja_version=%s\n' "$(ninja --version)"
    fi
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version | sed -n 's/.*release \([^,]*\).*/nvcc_release=\1/p'
    fi
    env | LC_ALL=C sort | grep -E '^(CMAKE_|SKBUILD_|USE_CUDA=|USE_ROCM=|USE_METAL=|TILELANG_USE_CUDA_STUBS=|WITH_PIP_CUDA_TOOLCHAIN=|CUDA_HOME=|CUDA_PATH=|CUDA_VERSION=|CUDACXX=|HIP_PATH=|ROCM_PATH=|NO_VERSION_LABEL=|NO_TOOLCHAIN_VERSION=|NO_GIT_VERSION=)' || true
}

build_wheel_to_dir() {
    local repo_dir="$1"
    local venv_dir="$2"
    local build_dir="$3"
    local wheel_dir="$4"

    rm -rf "${build_dir}"
    mkdir -p "$(dirname "${build_dir}")" "${wheel_dir}"

    if command -v uv >/dev/null 2>&1; then
        uv build -v --wheel --python "${venv_dir}/bin/python" -C "build-dir=${build_dir}" --out-dir "${wheel_dir}" "${repo_dir}"
    else
        "${venv_dir}/bin/python" -m pip wheel -v --no-deps -C "build-dir=${build_dir}" --wheel-dir "${wheel_dir}" "${repo_dir}"
    fi
}

install_wheel_file() {
    local venv_dir="$1"
    local wheel_file="$2"

    if [[ -z "${wheel_file}" || ! -f "${wheel_file}" ]]; then
        echo "Wheel file not found: ${wheel_file}" >&2
        exit 1
    fi

    if command -v uv >/dev/null 2>&1; then
        uv pip install -v "${wheel_file}"
    else
        "${venv_dir}/bin/python" -m pip install -v "${wheel_file}"
    fi
}

install_repo_wheel() {
    local label="$1"
    local repo_dir="$2"
    local venv_dir="$3"
    local build_dir="$4"

    if [[ "${WHEEL_CACHE}" != "1" ]]; then
        echo "Wheel cache disabled for ${label}."
        local uncached_dir="${build_dir}-wheel"
        rm -rf "${uncached_dir}"
        build_wheel_to_dir "${repo_dir}" "${venv_dir}" "${build_dir}" "${uncached_dir}"
        local uncached_wheel
        uncached_wheel="$(find "${uncached_dir}" -maxdepth 1 -type f -name '*.whl' | sort | tail -n 1)"
        install_wheel_file "${venv_dir}" "${uncached_wheel}"
        return
    fi

    if repo_has_untracked_files "${repo_dir}"; then
        echo "Wheel cache skipped for ${label}; untracked source files are present."
        local untracked_dir="${build_dir}-wheel"
        rm -rf "${untracked_dir}"
        build_wheel_to_dir "${repo_dir}" "${venv_dir}" "${build_dir}" "${untracked_dir}"
        local untracked_wheel
        untracked_wheel="$(find "${untracked_dir}" -maxdepth 1 -type f -name '*.whl' | sort | tail -n 1)"
        install_wheel_file "${venv_dir}" "${untracked_wheel}"
        return
    fi

    local metadata
    metadata="$(wheel_cache_metadata "${repo_dir}" "${venv_dir}/bin/python")"
    local cache_key
    cache_key="$(printf '%s\n' "${metadata}" | hash_stdin | cut -c 1-24)"
    local cache_entry="${WHEEL_CACHE_DIR%/}/${cache_key}"
    local cached_wheel
    cached_wheel="$(find "${cache_entry}" -maxdepth 1 -type f -name '*.whl' 2>/dev/null | sort | tail -n 1 || true)"

    mkdir -p "${WHEEL_CACHE_DIR}"
    echo "Wheel cache key for ${label}: ${cache_key}"
    echo "Wheel cache dir: ${WHEEL_CACHE_DIR}"

    if [[ "${REFRESH_WHEEL_CACHE}" != "1" && -n "${cached_wheel}" && -f "${cached_wheel}" ]]; then
        echo "Wheel cache hit for ${label}: ${cached_wheel}"
        install_wheel_file "${venv_dir}" "${cached_wheel}"
        return
    fi

    echo "Wheel cache miss for ${label}; building wheel."
    local tmp_entry="${cache_entry}.tmp.$$"
    rm -rf "${tmp_entry}"
    mkdir -p "${tmp_entry}"
    printf '%s\n' "${metadata}" >"${tmp_entry}/metadata.txt"
    build_wheel_to_dir "${repo_dir}" "${venv_dir}" "${build_dir}" "${tmp_entry}"

    local built_wheel
    built_wheel="$(find "${tmp_entry}" -maxdepth 1 -type f -name '*.whl' | sort | tail -n 1)"
    if [[ -z "${built_wheel}" || ! -f "${built_wheel}" ]]; then
        echo "No wheel produced for ${label}." >&2
        exit 1
    fi

    rm -rf "${cache_entry}"
    mv "${tmp_entry}" "${cache_entry}"
    cached_wheel="$(find "${cache_entry}" -maxdepth 1 -type f -name '*.whl' | sort | tail -n 1)"
    echo "Wheel cached for ${label}: ${cached_wheel}"
    install_wheel_file "${venv_dir}" "${cached_wheel}"
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
        else
            python -m pip install -v -r requirements-test.txt
        fi
        install_repo_wheel "${label}" "${repo_dir}" "${venv_dir}" "${build_dir}"
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
        run_isolated_python "${python_bin}" "${SCRIPT_DIR}/run_current_regression.py" \
            "${runner_flags[@]}" \
            "--examples-root" "${repo_dir}/examples" \
            "--use-installed-package" \
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
    run_isolated_python "${CURRENT_VENV}/bin/python" "${SCRIPT_DIR}/compare_perf_regression.py" "${compare_flags[@]}"
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
