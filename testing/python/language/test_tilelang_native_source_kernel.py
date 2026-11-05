"""Raw CUDA source keeps its native signature and launch identity."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tvm_tirx import ir


def _copy_source(entry, delta=0):
    return f"""
extern "C" __global__ void {entry}(float* A, float* Z, int n) {{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) Z[i] = A[i] + {delta}.0f;
}}
"""


def _reverse_arguments(entry="source_copy", delta=0):
    N = T.dynamic("N")
    source = _copy_source(entry, delta)

    @T.prim_func
    def main(Z: T.Tensor((N,), T.float32), A: T.Tensor((N,), T.float32)):
        T.CUDASourceCodeKernel(T.ceildiv(N, 128), threads=128, source_code_or_path=source, entry_name=entry)

    return main


def test_simplify_keeps_external_source_parameters():
    function = _reverse_arguments()
    simplified = tilelang.transform.Simplify(simplify_arguments=True)(ir.IRModule({"main": function}))["main"]
    assert [parameter.name for parameter in simplified.params] == ["Z", "A"]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("length", [1, 129, 257])
def test_source_parameters_keep_baseline_sorted_abi(length):
    # The public order is Z,A; the external device ABI is pointer/name sorted,
    # so it must still pass A,Z,N. Reversing a pointer silently corrupts output.
    function = _reverse_arguments(delta=3)
    artifact = tilelang.lower(function, target="cuda")
    symbol, device = next(iter(artifact.device_mod.functions.items()))
    assert symbol.name_hint == "source_copy"
    assert len(device.params) == 3
    assert str(device.attrs["global_symbol"]) == "source_copy"
    kernel = tilelang.compile(function, target="cuda", out_idx=0)
    a = torch.arange(length, dtype=torch.float32, device="cuda")
    z = kernel(a)
    torch.testing.assert_close(z, a + 3, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_source_signature_carries_both_dynamic_dimensions():
    N = T.dynamic("N")
    M = T.dynamic("M")
    source = """
extern "C" __global__ void source_matrix(float* A, float* Z, int m, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n * m) Z[i] = A[i] + (float)(10 * m + n);
}
"""

    @T.prim_func
    def main(Z: T.Tensor((N, M), T.float32), A: T.Tensor((N, M), T.float32)):
        T.CUDASourceCodeKernel(T.ceildiv(N * M, 128), threads=128, source_code_or_path=source, entry_name="source_matrix")

    kernel = tilelang.compile(main, target="cuda", out_idx=0)
    a = torch.arange(21, dtype=torch.float32, device="cuda").reshape(3, 7)
    torch.testing.assert_close(kernel(a), a + 73, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("first_entry", ["source_first", "main_kernel_1"])
def test_distinct_source_entries_preserve_host_launch_order(first_entry):
    N = T.dynamic("N")
    first = _copy_source(first_entry, 1)
    second = """
extern "C" __global__ void source_second(float* A, float* Z, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) Z[i] = 2.0f * Z[i];
}
"""

    @T.prim_func
    def main(A: T.Tensor((N,), T.float32), Z: T.Tensor((N,), T.float32)):
        T.CUDASourceCodeKernel(T.ceildiv(N, 128), threads=128, source_code_or_path=first, entry_name=first_entry)
        T.CUDASourceCodeKernel(T.ceildiv(N, 128), threads=128, source_code_or_path=second, entry_name="source_second")

    kernel = tilelang.compile(main, target="cuda", out_idx=1)
    a = torch.arange(257, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kernel(a), 2 * (a + 1), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_source_signature_retains_noncontiguous_stride_symbol():
    N = T.dynamic("N")
    S = T.dynamic("S")
    source = """
extern "C" __global__ void source_stride(float* A, float* Z, int n, int s) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) Z[i] = A[i * s];
}
"""

    @T.prim_func
    def main(A: T.StridedTensor((N,), (S,), T.float32), Z: T.Tensor((N,), T.float32)):
        T.CUDASourceCodeKernel(T.ceildiv(N, 128), threads=128, source_code_or_path=source, entry_name="source_stride")

    kernel = tilelang.compile(main, target="cuda", out_idx=1)
    a = torch.arange(258, dtype=torch.float32, device="cuda")[::2]
    assert not a.is_contiguous()
    torch.testing.assert_close(kernel(a), a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_duplicate_source_entry_is_rejected_before_device_compilation():
    N = T.dynamic("N")
    source = _copy_source("duplicate_source")

    @T.prim_func
    def main(A: T.Tensor((N,), T.float32), Z: T.Tensor((N,), T.float32)):
        T.CUDASourceCodeKernel(T.ceildiv(N, 128), threads=128, source_code_or_path=source, entry_name="duplicate_source")
        T.CUDASourceCodeKernel(T.ceildiv(N, 128), threads=128, source_code_or_path=source, entry_name="duplicate_source")

    with pytest.raises(ValueError, match="Duplicate CUDA source entry"):
        tilelang.lower(main, target="cuda")
