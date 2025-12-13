#!/bin/bash

# Set ROOT_DIR to the project root (two levels up from this script's directory)
ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)

# Change to the project root directory for local testing of changes
cd $ROOT_DIR

# Add the project root to PYTHONPATH so Python can find local modules
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# Run pytest in parallel (4 workers) for all tests in the examples directory
cd examples
python -m pytest -n 4 . --verbose --color=yes --durations=0 --showlocals --cache-clear
cd ..

# Run pytest in parallel (4 workers) for all tests in the testing/python directory.
# IMPORTANT: CuTeDSL backend currently requires GEMM v1 (TILELANG_USE_GEMM_V1=1).
# Do NOT export it globally here, or you'll silently change the default GEMM selection
# for unrelated tests. Run the CuTeDSL JIT tests in a separate pytest invocation.
cd testing/python
python -m pytest -n 4 . --ignore=jit/test_tilelang_jit_cutedsl.py --verbose --color=yes --durations=0 --showlocals --cache-clear

# CuTeDSL JIT tests (isolate env + avoid xdist contention on a single GPU)
TILELANG_USE_GEMM_V1=1 python -m pytest -n 1 jit/test_tilelang_jit_cutedsl.py --verbose --color=yes --durations=0 --showlocals --cache-clear
cd ..
