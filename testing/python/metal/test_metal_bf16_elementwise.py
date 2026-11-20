"""BF16 scalar constants and short vectors must retain their Metal types."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@tilelang.jit(target="metal", execution_backend="torch")
def bf16_elementwise(size, add=False):
    @T.prim_func
    def main(A: T.Tensor((size,), "bfloat16"), B: T.Tensor((size,), "bfloat16")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(size):
                B[i] = A[i] + T.bfloat16(1) if add else A[i]

    return main


@tilelang.jit(target="metal", execution_backend="torch")
def bf16_sliced_copy(lanes):
    # The vectorizer only widens power-of-two extents; explicit slices reach
    # the three-lane (packed) vector type as well.
    @T.prim_func
    def main(A: T.Tensor((128 * lanes,), "bfloat16"), B: T.Tensor((128 * lanes,), "bfloat16")):
        with T.Kernel(1, threads=128):
            tid = T.get_thread_binding(0)
            B[tid * lanes : tid * lanes + lanes] = A[tid * lanes : tid * lanes + lanes]

    return main


@tilelang.jit(target="metal", execution_backend="torch")
def bf16_fill(value):
    @T.prim_func
    def main(B: T.Tensor((128,), "bfloat16")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = T.bfloat16(value)

    return main


def _source(program):
    with tvm.target.Target("metal"):
        return tilelang.lower(program, target="metal").kernel_source


@pytest.mark.parametrize("size, metal_type", [(256, "bfloat2"), (512, "bfloat4"), (1024, "uint4")])
def test_bf16_copy_vector_type(size, metal_type):
    source = _source(bf16_elementwise.get_tir(size))
    assert f"device {metal_type}*" in source


@pytest.mark.parametrize("lanes, metal_type", [(2, "bfloat2"), (3, "packed_bfloat3"), (4, "bfloat4")])
def test_bf16_sliced_copy_vector_type(lanes, metal_type):
    source = _source(bf16_sliced_copy.get_tir(lanes))
    assert f"device {metal_type}*" in source


def test_bf16_constant_codegen():
    source = _source(bf16_elementwise.get_tir(512, add=True))
    assert "bfloat(1.000000e+00f)" in source
    assert "1.000000e+00h" not in source


@tilelang.testing.requires_metal
@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_bf16_copy_execution(size):
    a = torch.arange(size, dtype=torch.float32).to(torch.bfloat16)
    b = torch.full((size,), -1, dtype=torch.bfloat16, device="mps")
    bf16_elementwise(size)(a.to("mps"), b)
    torch.testing.assert_close(b.cpu(), a, rtol=0, atol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("lanes", [2, 3, 4])
def test_bf16_sliced_copy_execution(lanes):
    size = 128 * lanes
    a = torch.arange(size, dtype=torch.float32).to(torch.bfloat16)
    b = torch.full((size,), -1, dtype=torch.bfloat16, device="mps")
    bf16_sliced_copy(lanes)(a.to("mps"), b)
    torch.testing.assert_close(b.cpu(), a, rtol=0, atol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("size", [128, 256, 512])
def test_bf16_add_execution(size):
    a = torch.linspace(-2, 2, size).to(torch.bfloat16)
    b = torch.empty_like(a, device="mps")
    bf16_elementwise(size, add=True)(a.to("mps"), b)
    torch.testing.assert_close(b.cpu(), a + 1, rtol=0, atol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("value", [0.0, -0.0, 1.5, -1e30, float("inf"), -float("inf"), float("nan")])
def test_bf16_constant_execution(value):
    b = torch.empty(128, dtype=torch.bfloat16, device="mps")
    bf16_fill(value)(b)
    expected = torch.full((128,), value, dtype=torch.bfloat16)
    actual = b.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
