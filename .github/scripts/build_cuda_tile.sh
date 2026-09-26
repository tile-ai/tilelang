#!/usr/bin/env bash
# Build the public CUDA Tile 13.4 Python bindings using the active Python.
set -euo pipefail

# v13.4.0 and the LLVM revision selected by its cmake/IncludeLLVM.cmake.
CUDA_TILE_REV=7e8e2e68fa219716103824c01f7303367cf7df8d
LLVM_REV=9ebb067a8a2b4b0705f06d59c77d36dfab98333f
build_root="$(realpath -m "${1:?Usage: build_cuda_tile.sh BUILD_ROOT}")"
python_bin="$(command -v python)"

fetch_source() {
  local repository="$1" revision="$2" destination="$3"
  # Each CI run supplies a fresh directory; never reuse an unverified checkout.
  mkdir -p "${destination}"
  git -C "${destination}" init
  git -C "${destination}" fetch --depth=1 "${repository}" "${revision}"
  git -C "${destination}" checkout --detach FETCH_HEAD
  test "$(git -C "${destination}" rev-parse HEAD)" = "${revision}"
}

fetch_source https://github.com/NVIDIA/cuda-tile.git "${CUDA_TILE_REV}" "${build_root}/cuda-tile"
fetch_source https://github.com/llvm/llvm-project.git "${LLVM_REV}" "${build_root}/llvm-project"
uv pip install --python "${python_bin}" 'cmake>=3.26,<4' ninja \
  -r "${build_root}/llvm-project/mlir/python/requirements.txt"

cmake -G Ninja -S "${build_root}/cuda-tile" -B "${build_root}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON \
  -DCUDA_TILE_ENABLE_TESTING=OFF \
  -DCUDA_TILE_USE_LLVM_SOURCE_DIR="${build_root}/llvm-project" \
  -DPython3_EXECUTABLE="${python_bin}"
cmake --build "${build_root}/build" --target CudaTilePythonModules --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

bindings="${build_root}/build/python_packages"
PYTHONPATH="${bindings}" "${python_bin}" -c 'from cuda_tile._mlir.dialects import cuda_tile_ops; from cuda_tile._mlir import ir; print(cuda_tile_ops.__file__)'
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "PYTHONPATH=${bindings}" >> "${GITHUB_ENV}"
fi
