#!/bin/bash

# Usage:
#   bash maint/scripts/run_local_ci_test_ascend.sh
#
# What it does:
#   - Runs examples/ascend/ and testing/ascend/ with pytest-xdist, matching CI.
#   - Does NOT load the CUDA scheduler plugin and does NOT auto-detect GPUs.
#   - Defaults to 4 example workers and 32 unit-test workers, as in ci.yml.
#   - Stops on a failed suite; each suite stops after 3 failures.
#
# Environment variables:
#   - PYTEST_XDIST_WORKERS: override worker count for both suites.
#   - PYTEST_XDIST_UNIT_WORKERS: override unit-test workers independently.
#
# Examples:
#   - Default run:                 bash maint/scripts/run_local_ci_test_ascend.sh
#   - Use 8 workers for both:      PYTEST_XDIST_WORKERS=8 bash maint/scripts/run_local_ci_test_ascend.sh
#   - Limit unit-test workers:     PYTEST_XDIST_UNIT_WORKERS=8 bash maint/scripts/run_local_ci_test_ascend.sh
#
# Requirements:
#   - pytest, pytest-xdist
#   - bisheng / Ascend toolchain available in the environment

set -e

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
UNIT_NWORKERS=${PYTEST_XDIST_UNIT_WORKERS:-${PYTEST_XDIST_WORKERS:-32}}
if ! [[ "$UNIT_NWORKERS" =~ ^[0-9]+$ ]] || [[ "$UNIT_NWORKERS" -le 0 ]]; then
  UNIT_NWORKERS=32
fi

PYTEST_ARGS_COMMON=(--verbose --color=yes --durations=0 --showlocals --cache-clear --maxfail=3)

# Match the working directory and suite paths used by .github/workflows/ci.yml.
cd testing || exit 1
echo "[INFO] DEVICE=ascend; running example tests without CUDA plugin. Workers: $NWORKERS."
python -m pytest -n "$NWORKERS" ../examples/ascend "${PYTEST_ARGS_COMMON[@]}"

echo "[INFO] DEVICE=ascend; running unit tests without CUDA plugin. Workers: $UNIT_NWORKERS."
python -m pytest -n "$UNIT_NWORKERS" ascend "${PYTEST_ARGS_COMMON[@]}"
