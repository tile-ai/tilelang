"""Test Metal simdgroup register GEMM with direct simdgroup_store to device memory.

These tests verify the simdgroup register accumulation path where C is allocated
in metal.simdgroup scope. This eliminates C_simd load/store round-trips through
shared memory on each K iteration. The final T.copy(C_local, C[...]) is lowered
to simdgroup_store directly to device memory via LowerSIMDGroupStore.
"""

import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import torch


def _make_simdgroup_gemm_func(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


matmul_simdgroup = tilelang.jit(_make_simdgroup_gemm_func)


def assert_simdgroup_store_correctness(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32, atol=1e-2):
    kernel = matmul_simdgroup(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)

    torch_dtype = dtype.as_torch()
    torch_accum_dtype = accum_dtype.as_torch()
    a = torch.randn(M, K, dtype=torch_dtype, device="mps")
    b = torch.randn(K, N, dtype=torch_dtype, device="mps")
    c = torch.zeros(M, N, dtype=torch_accum_dtype, device="mps")

    kernel(a, b, c)

    ref = a.to(torch_accum_dtype) @ b.to(torch_accum_dtype)
    assert torch.allclose(ref, c, atol=atol), (
        f"Result mismatch for M={M}, N={N}, K={K}, "
        f"block=({block_M},{block_N},{block_K}), dtype={dtype}\n"
        f"max diff: {(ref - c).abs().max().item()}"
    )


def assert_simdgroup_store_codegen(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    func = _make_simdgroup_gemm_func(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")

    src = artifact.kernel_source
    assert src is not None
    assert "kernel void" in src
    assert "simdgroup_multiply_accumulate" in src
    assert "make_filled_simdgroup_matrix" in src

    assert "simdgroup_float8x8" in src or "simdgroup_half8x8" in src, "Expected simdgroup_float8x8 or simdgroup_half8x8 for C accumulator"

    store_to_device = src.count("simdgroup_store(C_local")
    assert store_to_device > 0, "Expected simdgroup_store of C_local to device memory"

    load_c_from_shared = [line for line in src.split("\n") if "simdgroup_load" in line and "C_local" in line]
    assert len(load_c_from_shared) == 0, f"C_local should not be loaded from shared memory, but found: {load_c_from_shared}"


# --- Codegen tests (cross-platform) ---


def test_codegen_square_small():
    assert_simdgroup_store_codegen(64, 64, 64, 16, 16, 16)


def test_codegen_square_large():
    assert_simdgroup_store_codegen(128, 128, 128, 32, 32, 32)


def test_codegen_non_square():
    assert_simdgroup_store_codegen(128, 128, 128, 32, 64, 16)


def test_codegen_float32_accum():
    assert_simdgroup_store_codegen(64, 64, 64, 16, 16, 16, dtype=T.float32, accum_dtype=T.float32)


# --- Correctness tests (require Metal hardware) ---


@tilelang.testing.requires_metal
def test_correctness_16x16x16():
    assert_simdgroup_store_correctness(128, 128, 128, 16, 16, 16)


@tilelang.testing.requires_metal
def test_correctness_32x32x32():
    assert_simdgroup_store_correctness(128, 128, 128, 32, 32, 32)


@tilelang.testing.requires_metal
def test_correctness_non_square_block():
    assert_simdgroup_store_correctness(128, 128, 128, 32, 64, 16)


@tilelang.testing.requires_metal
def test_correctness_64x64x32():
    assert_simdgroup_store_correctness(128, 128, 128, 64, 64, 32)


@tilelang.testing.requires_metal
def test_correctness_large_matrix():
    assert_simdgroup_store_correctness(1024, 1024, 1024, 32, 32, 32, atol=1.0)


@tilelang.testing.requires_metal
def test_correctness_non_square_matrix():
    assert_simdgroup_store_correctness(256, 512, 128, 32, 32, 16)


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
