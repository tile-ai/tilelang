import os
import re
import tempfile
from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing

import torch


CUDA_SOURCE = """
extern "C" __global__ void external_copy(float* A, float* B, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        B[i] = A[i];
    }
}
"""


def make_source_kernel(source_code_or_path: str | os.PathLike[str], entry_name: str):
    N = T.dynamic("N")

    @T.prim_func
    def main(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
    ):
        T.CUDASourceCodeKernel(T.ceildiv(N, 128), threads=128, source_code_or_path=source_code_or_path, entry_name=entry_name)

    return main


def get_single_device_function_name(device_mod) -> str:
    function_names = [g_var.name_hint for g_var in device_mod.functions]
    assert len(function_names) == 1
    return function_names[0]


@tilelang.testing.requires_cuda
def test_source_kernel_inline_codegen():
    artifact = tilelang.lower(make_source_kernel(CUDA_SOURCE, entry_name="external_copy"), target="cuda")
    function_name = get_single_device_function_name(artifact.device_mod)

    assert re.search(
        rf"__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?{re.escape(function_name)}\s*\(",
        artifact.kernel_source,
    )
    assert "B[i] = A[i];" in artifact.kernel_source


@tilelang.testing.requires_cuda
def test_source_kernel_run():
    kernel = tilelang.compile(make_source_kernel(CUDA_SOURCE, entry_name="external_copy"), target="cuda")
    print(kernel.get_kernel_source())
    print(kernel.get_host_source())
    a = torch.randn(128, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    kernel(a, b)
    torch.testing.assert_close(b, a)


@tilelang.testing.requires_cuda
def test_source_kernel_loads_from_file():
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False, encoding="utf-8") as f:
        f.write(CUDA_SOURCE)
        source_path = f.name

    try:
        artifact = tilelang.lower(make_source_kernel(Path(source_path), entry_name="external_copy"), target="cuda")
    finally:
        os.unlink(source_path)

    assert "B[i] = A[i];" in artifact.kernel_source


@tilelang.testing.requires_cuda
def test_source_kernel_invalid_entry_name_fails_in_lower():
    with pytest.raises(Exception, match=r"Available entries: external_copy"):
        tilelang.lower(make_source_kernel(CUDA_SOURCE, entry_name="main_kernel"), target="cuda")


if __name__ == "__main__":
    tilelang.testing.main()
