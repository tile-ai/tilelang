#!/usr/bin/env bash
# Build a portable matched bundle inside a manylinux environment; do not publish.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export NO_VERSION_LABEL=ON
TILELANG_NATIVE_STAGE="${TILELANG_NATIVE_STAGE:-.native-portable-stage}" \
  bash maint/scripts/local_distribution.sh raw_dist
mkdir -p dist
cp raw_dist/*.tar.gz dist/
for wheel in raw_dist/*.whl; do
  uv tool run --with patchelf --from auditwheel auditwheel repair \
    --exclude libtvm_ffi.so --exclude 'libz3.so*' --exclude libcuda.so.1 \
    --exclude '/usr/local/cuda*' --exclude '/opt/amdgpu*' --exclude '/opt/rocm*' \
    -w dist "${wheel}"
done
