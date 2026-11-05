#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
CIBW_BUILD='cp310-*' python maint/scripts/build_native_wheels.py \
  --stage .native-local-wheel-stage --output wheelhouse --jobs "${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
  2>&1 | tee cibuildwheel.log
