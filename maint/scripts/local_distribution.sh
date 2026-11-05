#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TILELANG_LOCAL_STAGE="${TILELANG_NATIVE_STAGE:-.native-local-stage}"
TILELANG_LOCAL_OUTPUT="${1:-dist}"
python maint/scripts/build_native_artifacts.py stage --destination "${TILELANG_LOCAL_STAGE}"
python maint/scripts/build_native_artifacts.py build \
  --stage "${TILELANG_LOCAL_STAGE}" --output "${TILELANG_LOCAL_OUTPUT}" \
  --frontend uv --jobs "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
