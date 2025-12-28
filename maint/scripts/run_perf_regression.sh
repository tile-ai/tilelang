#!/bin/bash
# Performance regression test: compare current branch vs origin/main
#
# Usage:
#   ./maint/scripts/run_perf_regression.sh
#
# Environment variables:
#   PYTHON_VERSION  - Python version to use (default: 3.12)
#   WORK_DIR        - Working directory for venvs (default: .perf_regression)
#   SKIP_BUILD_NEW  - Set to 1 to skip rebuilding new venv
#   SKIP_BUILD_OLD  - Set to 1 to skip rebuilding old venv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/.perf_regression}"
RESULT_MD="${WORK_DIR}/regression_result.md"
RESULT_PNG="${WORK_DIR}/regression_result.png"

UPSTREAM_URL="https://github.com/tile-ai/tilelang"

# Check if user has a local build directory that might conflict with pip install
BUILD_DIR="${REPO_ROOT}/build"
BUILD_BACKUP=""
if [[ -d "${BUILD_DIR}" ]]; then
    BUILD_BACKUP="${BUILD_DIR}.bak.$$"
    echo "Found existing build directory, will rename to ${BUILD_BACKUP}"
fi

echo "============================================"
echo "Performance Regression Test"
echo "============================================"
echo "Repo root:      ${REPO_ROOT}"
echo "Work dir:       ${WORK_DIR}"
echo "Python version: ${PYTHON_VERSION}"
echo "Upstream:       ${UPSTREAM_URL}"
echo ""

cd "${REPO_ROOT}"

# Ensure origin points to the correct upstream
ORIGIN_URL="$(git remote get-url origin 2>/dev/null || echo "")"
if [[ "${ORIGIN_URL}" != *"tile-ai/tilelang"* ]]; then
    echo "WARNING: origin (${ORIGIN_URL}) does not point to ${UPSTREAM_URL}"
    echo "Adding 'upstream' remote..."
    git remote remove upstream 2>/dev/null || true
    git remote add upstream "${UPSTREAM_URL}"
    REMOTE="upstream"
else
    REMOTE="origin"
fi
echo "Using remote: ${REMOTE}"

# Check for uncommitted changes
if [[ -n "$(git status --porcelain)" ]]; then
    echo "WARNING: You have uncommitted changes. They will be stashed."
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    STASHED=1
    git stash push -m "perf_regression_temp_stash"
else
    STASHED=0
fi

# Save current branch/commit
CURRENT_REF="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${CURRENT_REF}" == "HEAD" ]]; then
    # Detached HEAD
    CURRENT_REF="$(git rev-parse HEAD)"
fi
echo "Current ref: ${CURRENT_REF}"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    cd "${REPO_ROOT}"
    git checkout "${CURRENT_REF}" 2>/dev/null || true
    git submodule update --init --recursive 2>/dev/null || true
    if [[ "${STASHED}" == "1" ]]; then
        echo "Restoring stashed changes..."
        git stash pop || true
    fi
    # Restore original build directory if it was backed up
    if [[ -n "${BUILD_BACKUP}" && -d "${BUILD_BACKUP}" ]]; then
        echo "Restoring original build directory..."
        rm -rf "${BUILD_DIR}" 2>/dev/null || true
        mv "${BUILD_BACKUP}" "${BUILD_DIR}"
    fi
}
trap cleanup EXIT

# Create work directory
mkdir -p "${WORK_DIR}"

# Fetch latest main from upstream
echo ""
echo "Fetching ${REMOTE}/main..."
git fetch "${REMOTE}" main

# Backup existing build directory to avoid conflict with pip install
if [[ -n "${BUILD_BACKUP}" ]]; then
    echo "Renaming build -> ${BUILD_BACKUP##*/}"
    mv "${BUILD_DIR}" "${BUILD_BACKUP}"
fi

# ============================================
# Build NEW version (current branch)
# ============================================
if [[ "${SKIP_BUILD_NEW}" != "1" ]]; then
    echo ""
    echo "============================================"
    echo "Building NEW version (current branch: ${CURRENT_REF})"
    echo "============================================"

    git checkout "${CURRENT_REF}"
    git submodule update --init --recursive

    rm -rf "${WORK_DIR}/new"
    uv venv --python "${PYTHON_VERSION}" "${WORK_DIR}/new"
    source "${WORK_DIR}/new/bin/activate"

    uv pip install -v -r requirements-test.txt
    uv pip install -v .

    deactivate
else
    echo "Skipping NEW build (SKIP_BUILD_NEW=1)"
fi

# ============================================
# Build OLD version (upstream main)
# ============================================
if [[ "${SKIP_BUILD_OLD}" != "1" ]]; then
    echo ""
    echo "============================================"
    echo "Building OLD version (${REMOTE}/main)"
    echo "============================================"

    # Clean build artifacts before switching
    # Note: -e requires relative paths, not absolute paths
    git clean -dxf -e .perf_regression/ -e .cache/ -e "*.egg-info" -e "build.bak.*"

    git checkout "${REMOTE}/main"
    git submodule update --init --recursive

    rm -rf "${WORK_DIR}/old"
    uv venv --python "${PYTHON_VERSION}" "${WORK_DIR}/old"
    source "${WORK_DIR}/old/bin/activate"

    uv pip install -v -r requirements-test.txt
    uv pip install -v .

    deactivate
else
    echo "Skipping OLD build (SKIP_BUILD_OLD=1)"
fi

# ============================================
# Run regression test
# ============================================
echo ""
echo "============================================"
echo "Running performance regression test"
echo "============================================"

# Switch back to current branch for running the test script
git checkout "${CURRENT_REF}"
git submodule update --init --recursive

source "${WORK_DIR}/new/bin/activate"

OLD_PYTHON="${WORK_DIR}/old/bin/python" \
NEW_PYTHON="${WORK_DIR}/new/bin/python" \
PERF_REGRESSION_MD="${RESULT_MD}" \
PERF_REGRESSION_PNG="${RESULT_PNG}" \
    python "${SCRIPT_DIR}/test_perf_regression.py"

deactivate

echo ""
echo "============================================"
echo "Results"
echo "============================================"
echo "Markdown: ${RESULT_MD}"
echo "Plot:     ${RESULT_PNG}"
echo ""
cat "${RESULT_MD}"
