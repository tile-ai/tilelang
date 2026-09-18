"""Regression coverage for vectorized integer bitwise NOT (#2993)."""

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.cuda import language as T


DTYPES = [f"{sign}{bits}" for sign in ("int", "uint") for bits in (8, 16, 32, 64)]


def make_bitwise_not(dtype, elements_per_thread):
    n = 32 * elements_per_thread

    @T.prim_func
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(n):
                B[i] = ~A[i]

    return kernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("elements_per_thread", [1, 2, 4])
def test_bitwise_not_codegen(dtype, elements_per_thread):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        source = tilelang.lower(make_bitwise_not(dtype, elements_per_thread), target=target).kernel_source
    assert "~" in source
    if elements_per_thread > 1:
        # A direct complement of a packed vector load is invalid for CUDA structs.
        assert "(~*(" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("elements_per_thread", [1, 2, 4])
def test_bitwise_not_runtime(dtype, elements_per_thread):
    bits = int(dtype.removeprefix("uint").removeprefix("int"))
    mask = (1 << bits) - 1
    signed = dtype.startswith("int")

    def as_value(value):
        return value - (1 << bits) if signed and value >= (1 << (bits - 1)) else value

    patterns = [0, mask, 1, 1 << (bits - 1), (1 << (bits - 1)) - 1, mask // 3, 2 * (mask // 3), mask - 1]
    values = patterns * (32 * elements_per_thread // len(patterns))
    torch_dtype = getattr(torch, dtype)
    a = torch.tensor([as_value(x) for x in values], dtype=torch_dtype, device="cuda")
    expected = [as_value(x ^ mask) for x in values]
    kernel = tilelang.compile(
        make_bitwise_not(dtype, elements_per_thread),
        out_idx=[1],
        target="cuda",
        execution_backend="tvm_ffi",
    )
    assert kernel(a).cpu().tolist() == expected
