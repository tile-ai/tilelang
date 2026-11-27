"""Ordinary TileLang operations on SIMD-group GEMM accumulators."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@tilelang.jit(target="metal", execution_backend="torch")
def fragment_gemm(m, n, k, policy, clear, repeat=False):
    @T.prim_func
    def main(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((m, k), "float16")
            b = T.alloc_shared((k, n), "float16")
            c = T.alloc_fragment((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            # An index-dependent initial value checks the native lane mapping.
            for i, j in T.Parallel(m, n):
                c[i, j] = T.float32(i * n + j) / 32
            T.gemm(a, b, c, clear_accum=clear, policy=policy)
            for i, j in T.Parallel(m, n):
                c[i, j] = c[i, j] * 0.5 + T.float32(j)
            if repeat:
                T.gemm(a, b, c, policy=policy)
            T.copy(c, C)

    return main


def test_fragment_access_codegen():
    program = fragment_gemm.get_tir(32, 32, 16, T.GemmWarpPolicy.Square, False)
    with tvm.target.Target("metal"):
        source = tilelang.lower(program, target="metal").kernel_source
    assert "simdgroup_multiply_accumulate" in source
    assert ".thread_elements()" in source


@tilelang.testing.requires_metal
@pytest.mark.parametrize("m,n,k", [(32, 32, 16), (32, 64, 32), (64, 32, 16)])
@pytest.mark.parametrize("policy", [T.GemmWarpPolicy.Square, T.GemmWarpPolicy.FullRow, T.GemmWarpPolicy.FullCol])
@pytest.mark.parametrize("clear,repeat", [(True, False), (False, False), (False, True)])
def test_fragment_access_execution(m, n, k, policy, clear, repeat):
    a = torch.randn(m, k, dtype=torch.float16)
    b = torch.randn(k, n, dtype=torch.float16)
    c = torch.empty(m, n, device="mps")
    fragment_gemm(m, n, k, policy, clear, repeat)(a.to("mps"), b.to("mps"), c)
    product = a.float() @ b.float()
    initial = 0 if clear else torch.arange(m * n).reshape(m, n) / 32
    expected = (product + initial) * 0.5 + torch.arange(n)
    if repeat:
        expected += product
    torch.testing.assert_close(c.cpu(), expected, atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_metal
def test_fragment_scalar_access():
    program = fragment_gemm.get_tir(32, 32, 16, T.GemmWarpPolicy.Square, True)
    kernel = tilelang.compile(
        program, target="metal", execution_backend="torch", pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True}
    )
    a = torch.randn(32, 16, dtype=torch.float16)
    b = torch.randn(16, 32, dtype=torch.float16)
    c = torch.empty(32, 32, device="mps")
    kernel(a.to("mps"), b.to("mps"), c)
    torch.testing.assert_close(c.cpu(), (a.float() @ b.float()) * 0.5 + torch.arange(32), atol=1e-4, rtol=1e-4)
