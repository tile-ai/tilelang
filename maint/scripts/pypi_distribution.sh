#!/usr/bin/env bash

set -eux

rm -rf dist raw_dist

python -mpip install -U pip
python -mpip install -U build wheel auditwheel patchelf

export NO_VERSION_LABEL=1

python -m build --sdist -o dist
python -m build --wheel -o raw_dist

auditwheel repair -L /lib -w dist \
    --exclude libcuda.so.1 --exclude /usr/local/cuda\* --exclude /opt/amdgpu\* \
    --exclude /opt/rocm\* \
    raw_dist/*.whl

echo "Wheel built successfully."
