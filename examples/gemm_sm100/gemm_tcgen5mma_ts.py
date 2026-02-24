"""
Chained GEMM example using tcgen05.st + MMA TS on Blackwell (SM100).

Test 1: SS GEMM + ld + cast (baseline correctness check)
Test 2: Full chained GEMM: SS → tcgen05.ld → cast → tcgen05.st → MMA TS → output
"""

import torch
import tilelang
import tilelang.language as T


def test_cast_only(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
):
    """Test 1: SS GEMM + ld + cast → output bf16"""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            P_local = T.alloc_fragment((block_M, block_N), in_dtype)
            P_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_tmem, transpose_A=False, transpose_B=True, mbar=mbar, wg_wait=-1, clear_accum=k == 0)
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, P_local)
            T.copy(P_local, P_shared)
            T.copy(P_shared, C[by * block_M, bx * block_N])

    return main


def chained_gemm(
    M,
    N1,
    N2,
    K,
    block_M,
    block_N1,
    block_N2,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
):
    """Test 2: Full chained GEMM with tcgen05.st"""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B1: T.Tensor((N1, K), in_dtype),
        B2: T.Tensor((N2, N1), in_dtype),
        D: T.Tensor((M, N2), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N2, block_N2), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B1_shared = T.alloc_shared((block_N1, block_K), in_dtype)
            S_tmem = T.alloc_tmem([block_M, block_N1], accum_dtype)
            mbar1 = T.alloc_barrier(1)

            S_local = T.alloc_fragment((block_M, block_N1), accum_dtype)
            P_local = T.alloc_fragment((block_M, block_N1), in_dtype)
            P_tmem = T.alloc_tmem([block_M, block_N1], in_dtype)

            B2_shared = T.alloc_shared((block_N2, block_N1), in_dtype)
            D_tmem = T.alloc_tmem([block_M, block_N2], accum_dtype)
            mbar2 = T.alloc_barrier(1)

            D_local = T.alloc_fragment((block_M, block_N2), accum_dtype)
            D_shared = T.alloc_shared((block_M, block_N2), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B1[0, k * block_K], B1_shared)
                T.gemm(A_shared, B1_shared, S_tmem, transpose_A=False, transpose_B=True, mbar=mbar1, wg_wait=-1, clear_accum=k == 0)
                T.mbarrier_wait_parity(mbar1, k % 2)

            T.copy(S_tmem, S_local)
            T.copy(S_local, P_local)
            T.copy(P_local, P_tmem)

            T.copy(B2[bx * block_N2, 0], B2_shared)
            T.gemm(P_tmem, B2_shared, D_tmem, transpose_A=False, transpose_B=True, mbar=mbar2, wg_wait=-1, clear_accum=True)
            T.mbarrier_wait_parity(mbar2, 0)

            T.copy(D_tmem, D_local)
            T.copy(D_local, D_shared)
            T.copy(D_shared, D[by * block_M, bx * block_N2])

    return main


PASS_CFG = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def run_test(name, func, out_idx, inputs, ref, rtol=1e-2, atol=1e-2):
    """Compile *func*, execute it with *inputs*, and assert closeness to *ref*."""
    jit = tilelang.compile(func, out_idx=out_idx, target="cuda", pass_configs=PASS_CFG)
    out = jit(*inputs).cpu()
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    print(f"{name}: PASSED")


if __name__ == "__main__":
    M, N, K = 128, 128, 128
    block_M, block_N, block_K = 128, 128, 128
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    threads = 128

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    ref_bf16 = (a.cpu().float() @ b.cpu().float().T).to(torch.bfloat16)

    # --- Test 1: SS GEMM + ld + cast (baseline) ---
    f1 = test_cast_only(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, threads)
    run_test("Test1 (SS+ld+cast)", f1, [2], [a, b], ref_bf16)

    # --- Test 2: Full chained GEMM (SS + st + TS) ---
    b2 = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    f2 = chained_gemm(M, N, N, K, block_M, block_N, block_N, block_K, in_dtype, out_dtype, accum_dtype, threads)
    ref_s = a.cpu().float() @ b.cpu().float().T
    ref_p = ref_s.to(torch.bfloat16).float()
    ref_d = (ref_p @ b2.cpu().float().T).to(torch.bfloat16)
    run_test("Test2 (chained GEMM)", f2, [3], [a, b, b2], ref_d)
