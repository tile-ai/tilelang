"""Tests for gfx950 (MI350) copy.async feature.

Two new behaviours introduced in commit dfa63b10:
  1. cp_async_gs<16> on gfx950 lowers to buffer_load_dwordx4 ... lds
     (128-bit direct-to-LDS, bypassing VGPRs) instead of a plain uint4
     scalar store.  coalesced_width=8 (8 fp16 = 16 bytes) is required to
     trigger the 16-byte path.
  2. CDNA arch helper reports smem_cap = 160 KB for gfx950, even when
     an older driver reports the conservative 64 KB default.
"""

import pytest
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.testing import _check_is_gfx950 as _is_gfx950


def _matmul_kernel(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads=128,
    k_pack=1,
    # coalesced_width=8 → cp_async_gs<16> (16 bytes, 8×fp16)
    # coalesced_width=4 → cp_async_gs<8>  (8 bytes,  4×fp16)
    coalesced_width=4,
):
    """Return a prim_func for pipelined GEMM using T.copy (global->shared)."""
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared, coalesced_width=coalesced_width)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=coalesced_width)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared, coalesced_width=coalesced_width)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared, coalesced_width=coalesced_width)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B, k_pack=k_pack)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# Test 1: codegen — cp_async_gs<16> is present in generated HIP source
# ---------------------------------------------------------------------------


def _matmul_kernel_async(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    coalesced_width=4,
    threads=128,
):
    """Return a prim_func using T.async_copy (explicit async, no pipeline) for codegen tests."""
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                if trans_A:
                    T.async_copy(A[k * block_K, by * block_M], A_shared, coalesced_width=coalesced_width)
                else:
                    T.async_copy(A[by * block_M, k * block_K], A_shared, coalesced_width=coalesced_width)
                if trans_B:
                    T.async_copy(B[bx * block_N, k * block_K], B_shared, coalesced_width=coalesced_width)
                else:
                    T.async_copy(B[k * block_K, bx * block_N], B_shared, coalesced_width=coalesced_width)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.testing.requires_gfx950
def test_gfx950_cp_async_gs_16_in_codegen():
    """coalesced_width=8 (16 bytes) must emit cp_async_gs<16> in generated HIP source.

    Uses T.async_copy (explicit async semantics) so that the 16-byte path is
    emitted verbatim without relying on the software pipeline planner.
    Pipelining correctness is covered separately by test_gfx950_copy_async_gemm_pipelined.
    """
    prog = _matmul_kernel_async(
        256,
        256,
        256,
        128,
        128,
        32,
        False,
        True,
        T.float16,
        T.float32,
        T.float32,
        coalesced_width=8,  # 8 fp16 = 16 bytes → cp_async_gs<16>
    )
    kernel = tl.compile(prog, out_idx=[2])
    src = kernel.get_kernel_source()
    assert "cp_async_gs<16>" in src, "Expected cp_async_gs<16> in generated HIP source for 128-bit async copy path"


# ---------------------------------------------------------------------------
# Test 2: LDS capacity reported as 160 KB on gfx950
# ---------------------------------------------------------------------------


@tilelang.testing.requires_rocm
def test_gfx950_smem_cap_160kb():
    """CDNA arch helper must report 160 KB LDS for gfx950."""
    from tilelang import tvm
    from tilelang.carver.arch.cdna import CDNA, _GFX950_LDS_SIZE

    target = tvm.target.Target("rocm")
    arch = CDNA(target)

    if _is_gfx950():
        assert arch.smem_cap == _GFX950_LDS_SIZE, f"Expected smem_cap={_GFX950_LDS_SIZE} for gfx950, got {arch.smem_cap}"
    else:
        # On non-gfx950 devices the override must NOT kick in
        from tilelang import tvm as _tvm

        dev = _tvm.device("rocm", 0)
        assert arch.smem_cap == dev.max_shared_memory_per_block


# ---------------------------------------------------------------------------
# Test 3: numerical correctness — pipelined copy.async GEMM (num_stages=2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trans_A, trans_B, k_pack",
    [
        (False, False, 1),
        (False, True, 1),
        (True, True, 1),
        (True, False, 1),
    ],
)
@tilelang.testing.requires_gfx950
def test_gfx950_copy_async_gemm_pipelined(trans_A, trans_B, k_pack):
    """Pipelined GEMM (num_stages=2) with gfx950 copy.async must be numerically correct."""
    prog = _matmul_kernel(
        512,
        512,
        512,
        128,
        128,
        32,
        trans_A,
        trans_B,
        T.float16,
        T.float32,
        T.float32,
        num_stages=2,
        threads=128,
        k_pack=k_pack,
        coalesced_width=4 * k_pack,
    )
    kernel = tl.compile(prog, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        a = A.T.float() if trans_A else A.float()
        b = B.T.float() if trans_B else B.float()
        return torch.matmul(a, b)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 4: non-pipelined baseline still correct (num_stages=0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trans_A, trans_B",
    [
        (False, False),
        (False, True),
        (True, True),
        (True, False),
    ],
)
@tilelang.testing.requires_rocm
def test_gfx950_copy_async_gemm_no_pipeline(trans_A, trans_B):
    """Non-pipelined GEMM (num_stages=0) must also produce correct results."""
    prog = _matmul_kernel(
        512,
        512,
        512,
        128,
        128,
        32,
        trans_A,
        trans_B,
        T.float16,
        T.float32,
        T.float32,
        num_stages=0,
        threads=128,
        coalesced_width=4,
    )
    kernel = tl.compile(prog, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        a = A.T.float() if trans_A else A.float()
        b = B.T.float() if trans_B else B.float()
        return torch.matmul(a, b)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
