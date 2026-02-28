"""Tests for tcgen05.st (Register → TMEM Store) and MMA TS on Blackwell (SM100).

Validates the chained GEMM pattern: SS GEMM → tcgen05.ld → cast → tcgen05.st → MMA TS,
which is the core building block for Flash Attention on Blackwell.
"""

import torch
import tilelang
import tilelang.testing
import tilelang.language as T

tilelang.testing.set_random_seed(0)

PASS_CFG = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def matmul_ss(M, N, K, bM, bN, bK, in_dtype, out_dtype, accum_dtype, threads):
    """SS GEMM baseline: verifies tcgen05.ld + cast pipeline works."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((bM, bK), in_dtype)
            B_shared = T.alloc_shared((bN, bK), in_dtype)
            C_tmem = T.alloc_tmem((bM, bN), accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((bM, bN), accum_dtype)
            C_local_cast = T.alloc_fragment((bM, bN), in_dtype)
            C_shared = T.alloc_shared((bM, bN), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=1):
                T.copy(A[by * bM, k * bK], A_shared)
                T.copy(B[bx * bN, k * bK], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    transpose_B=True,
                    mbar=mbar,
                    wg_wait=-1,
                    clear_accum=k == 0,
                )
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_local_cast)
            T.copy(C_local_cast, C_shared)
            T.copy(C_shared, C[by * bM, bx * bN])

    return main


def chained_gemm(
    M,
    N1,
    N2,
    K,
    bM,
    bN1,
    bN2,
    bK,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
):
    """Chained GEMM: SS → tcgen05.ld → cast → tcgen05.st → MMA TS → output."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B1: T.Tensor((N1, K), in_dtype),
        B2: T.Tensor((N2, N1), in_dtype),
        D: T.Tensor((M, N2), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N2, bN2), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((bM, bK), in_dtype)
            B1_shared = T.alloc_shared((bN1, bK), in_dtype)
            S_tmem = T.alloc_tmem([bM, bN1], accum_dtype)
            mbar1 = T.alloc_barrier(1)

            S_local = T.alloc_fragment((bM, bN1), accum_dtype)
            P_local = T.alloc_fragment((bM, bN1), in_dtype)
            P_tmem = T.alloc_tmem([bM, bN1], in_dtype)

            B2_shared = T.alloc_shared((bN2, bN1), in_dtype)
            D_tmem = T.alloc_tmem([bM, bN2], accum_dtype)
            mbar2 = T.alloc_barrier(1)

            D_local = T.alloc_fragment((bM, bN2), accum_dtype)
            D_shared = T.alloc_shared((bM, bN2), out_dtype)

            # Stage 1: SS GEMM -- S_tmem = A * B1^T (fp32 accumulator)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=1):
                T.copy(A[by * bM, k * bK], A_shared)
                T.copy(B1[0, k * bK], B1_shared)
                T.gemm(
                    A_shared,
                    B1_shared,
                    S_tmem,
                    transpose_B=True,
                    mbar=mbar1,
                    wg_wait=-1,
                    clear_accum=k == 0,
                )
                T.mbarrier_wait_parity(mbar1, k % 2)

            # tcgen05.ld (fp32) → cast to bf16 → tcgen05.st (bf16, packed)
            T.copy(S_tmem, S_local)
            T.copy(S_local, P_local)
            T.copy(P_local, P_tmem)

            # Stage 2: MMA TS -- D_tmem = P_tmem * B2^T
            T.copy(B2[bx * bN2, 0], B2_shared)
            T.gemm(
                P_tmem,
                B2_shared,
                D_tmem,
                transpose_B=True,
                mbar=mbar2,
                wg_wait=-1,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(mbar2, 0)

            T.copy(D_tmem, D_local)
            T.copy(D_local, D_shared)
            T.copy(D_shared, D[by * bM, bx * bN2])

    return main


def _cpu_ref_chained(a, b1, b2):
    """Compute chained GEMM reference on CPU to avoid cuBLAS issues on Blackwell."""
    s = a.cpu().float() @ b1.cpu().float().T
    p = s.to(torch.bfloat16).float()
    d = (p @ b2.cpu().float().T).to(torch.bfloat16)
    return d


def assert_ss_gemm(M, N, K, bM, bN, bK, threads=128):
    """Compile and run an SS GEMM, asserting correctness against a CPU reference."""
    func = matmul_ss(M, N, K, bM, bN, bK, T.bfloat16, T.bfloat16, T.float32, threads)
    kernel = tilelang.compile(func, out_idx=-1, target="cuda", pass_configs=PASS_CFG)

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    ref = (a.cpu().float() @ b.cpu().float().T).to(torch.bfloat16)

    out = kernel(a, b).cpu()
    tilelang.testing.torch_assert_close(out, ref, rtol=1e-2, atol=1e-2)


def assert_chained_gemm(M, N1, N2, K, bM, bN1, bN2, bK, threads=128):
    """Compile and run a chained GEMM (SS + TS), verifying tcgen05.st and mma_ts presence."""
    assert bN1 == N1, f"bN1 must equal N1 (full row tile) for chained GEMM, got bN1={bN1}, N1={N1}"
    func = chained_gemm(M, N1, N2, K, bM, bN1, bN2, bK, T.bfloat16, T.bfloat16, T.float32, threads)
    kernel = tilelang.compile(func, out_idx=-1, target="cuda", pass_configs=PASS_CFG)

    src = kernel.get_kernel_source()
    assert src is not None
    assert "tcgen05mma_ts" in src, "Expected tcgen05mma_ts in generated code"
    assert "tcgen05_st" in src, "Expected tcgen05_st in generated code"

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b1 = torch.randn(N1, K, device="cuda", dtype=torch.bfloat16)
    b2 = torch.randn(N2, N1, device="cuda", dtype=torch.bfloat16)
    ref = _cpu_ref_chained(a, b1, b2)

    out = kernel(a, b1, b2).cpu()
    tilelang.testing.torch_assert_close(out, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
def test_ss_gemm_bf16_baseline():
    assert_ss_gemm(128, 128, 128, 128, 128, 128)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
def test_chained_gemm_128():
    assert_chained_gemm(128, 128, 128, 128, 128, 128, 128, 128)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
def test_chained_gemm_256():
    assert_chained_gemm(256, 128, 128, 256, 128, 128, 128, 128)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
def test_chained_gemm_non_power_of_2_k():
    assert_chained_gemm(128, 128, 128, 192, 128, 128, 128, 64)


if __name__ == "__main__":
    test_ss_gemm_bf16_baseline()
    test_chained_gemm_128()
    test_chained_gemm_256()
    test_chained_gemm_non_power_of_2_k()
