"""Compile the guarded vector access path with NVRTC's device-only headers."""

import pytest


@pytest.mark.parametrize("standard", ["c++17", "c++20"])
def test_nvrtc_vector_memory_headers_and_pointer_width(standard):
    pytest.importorskip("cuda.bindings.nvrtc")
    from tilelang.contrib.nvrtc import compile_cuda
    from tilelang.env import CUDA_HOME, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
    from tilelang.jit.adapter.nvrtc.include_paths import discover_cuda_include_paths

    source = r"""
#include <tl_templates/cuda/vector_memory.h>
static_assert(sizeof(uintptr_t) == sizeof(void*));
static_assert(sizeof(ptrdiff_t) == sizeof(void*));
extern "C" __global__ void copy_vector(const unsigned char* src,
                                      unsigned char* dst, ptrdiff_t offset) {
  auto value = tl::load_vector<uint4, 16>(src + offset);
  tl::store_vector<uint4, 16>(dst + offset, value);
}
"""
    paths = [TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR, *discover_cuda_include_paths(CUDA_HOME)]
    ptx = compile_cuda(source, arch=80, options=[f"-std={standard}", *(f"-I{path}" for path in paths)], verbose=True)
    text = ptx.decode("utf-8")
    assert ".entry copy_vector" in text
    assert "ld.volatile" in text and "st.volatile" in text
