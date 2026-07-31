"""Regression tests for type-driven Metal pointer address spaces."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


M = N = K = 128
BLOCK_M = BLOCK_N = 16
BLOCK_K = 8
THREADS = 64


def shared_alias_gemm():
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(N, BLOCK_N),
            T.ceildiv(M, BLOCK_M),
            threads=THREADS,
        ) as (bx, by):
            As = T.alloc_shared((BLOCK_M, BLOCK_K), T.float16)
            Bs = T.alloc_shared((BLOCK_K, BLOCK_N), T.float16)
            Cl = T.alloc_shared((BLOCK_M, BLOCK_N), T.float32)

            T.clear(Cl)
            for k in T.serial(T.ceildiv(K, BLOCK_K)):
                T.copy(A[by * BLOCK_M, k * BLOCK_K], As)
                T.copy(B[k * BLOCK_K, bx * BLOCK_N], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * BLOCK_M, bx * BLOCK_N])

    return kernel


def lower_to_metal(enable_device_compile: bool) -> str:
    target = tvm.target.Target("metal", tvm.target.Target("llvm"))
    with target:
        artifact = tilelang.lower(
            shared_alias_gemm(),
            target=target,
            target_host="llvm",
            enable_host_codegen=False,
            enable_device_compile=enable_device_compile,
        )
    return artifact.kernel_source or ""


def test_shared_alias_address_spaces_are_type_driven(monkeypatch):
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    compiled_source = lower_to_metal(enable_device_compile=True)
    source_only = lower_to_metal(enable_device_compile=False)

    assert compiled_source == source_only
    assert "threadgroup half* As = (threadgroup half*)" in compiled_source
    assert "threadgroup half* Bs = (threadgroup half*)" in compiled_source
    assert "*(threadgroup half2*)(As +" in compiled_source
    assert "*(threadgroup half2*)(Bs +" in compiled_source
    assert "(half*)As" not in compiled_source
    assert "(half*)Bs" not in compiled_source


@tilelang.testing.requires_metal
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "torch"])
def test_shared_alias_address_spaces_execute_on_metal(execution_backend):
    kernel = tilelang.compile(
        shared_alias_gemm(),
        target="metal",
        execution_backend=execution_backend,
    )

    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    c = torch.zeros(M, N, dtype=torch.float32, device="mps")

    kernel(a, b, c)
    torch.mps.synchronize()

    expected = a.float() @ b.float()
    assert torch.allclose(c, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
