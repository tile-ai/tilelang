#!/bin/bash

# Usage:
#   bash maint/scripts/run_local_ci_test_ascend.sh
#
# What it does:
#   - Runs only the Ascend NPU tests under examples/ascend/ with pytest-xdist.
#   - Does NOT load the CUDA scheduler plugin and does NOT auto-detect GPUs.
#   - Parallelism is controlled via $PYTEST_XDIST_WORKERS (default 4 if unset).
#
# Environment variables:
#   - PYTEST_XDIST_WORKERS: total workers, default 4.
#
# Examples:
#   - Default run:                 bash maint/scripts/run_local_ci_test_ascend.sh
#   - Increase parallelism:        PYTEST_XDIST_WORKERS=8 bash maint/scripts/run_local_ci_test_ascend.sh
#
# Requirements:
#   - pytest, pytest-xdist
#   - bisheng / Ascend toolchain available in the environment

# Set ROOT_DIR to the project root (two levels up from this script's directory)
ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)

# Change to the project root directory for local testing of changes
cd "$ROOT_DIR" || exit 1

# Add the project root and plugin directory to PYTHONPATH so Python can find local modules
export PYTHONPATH=$ROOT_DIR:$ROOT_DIR/maint/scripts:$PYTHONPATH

# Worker count (no device detection for Ascend)
NWORKERS=${PYTEST_XDIST_WORKERS:-4}
if ! [[ "$NWORKERS" =~ ^[0-9]+$ ]] || [[ "$NWORKERS" -le 0 ]]; then
  NWORKERS=4
fi
echo "[INFO] DEVICE=ascend; running without CUDA plugin. Workers: $NWORKERS."

PYTEST_ARGS_COMMON=(--verbose --color=yes --durations=0 --showlocals --cache-clear)
PYTEST_ARGS_DEVICE=(-n "$NWORKERS")

# Run pytest in parallel for all tests in the examples/ascend directory
cd examples/ascend || exit 1
python -m pytest "${PYTEST_ARGS_DEVICE[@]}" . "${PYTEST_ARGS_COMMON[@]}"
