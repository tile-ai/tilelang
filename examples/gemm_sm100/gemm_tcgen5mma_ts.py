"""
Example: TCGEN5MMA TS variant — A from Tensor Memory, B from Shared Memory.

Demonstrates chained MMA: the first GEMM (SS) produces C in TMEM,
then the second GEMM (TS) reads that TMEM result as its A operand.

    D_tmem = (A_smem × B1_smem) × B2_smem
"""
import torch
import tilelang
import tilelang.language as T


def chained_matmul(
    M,
    N1,
    K,
    N2,
    block_M,
    block_N1,
    block_K,
    block_N2,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
):
    """Two-stage chained matmul using SS then TS variant.

    Stage 1 (SS): C_tmem[M, N1] = A_smem[M, K] × B1_smem[N1, K]^T
    Stage 2 (TS): D_tmem[M, N2] = C_tmem[M, N1] × B2_smem[N2, N1]^T

    For stage 2, C_tmem (already in Tensor Memory) is used directly as the
    A operand via the TS (TMEM-Shared) variant of tcgen05.mma.
    """

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
            C_tmem = T.alloc_tmem([block_M, block_N1], accum_dtype)
            mbar1 = T.alloc_barrier(1)

            B2_shared = T.alloc_shared((block_N2, block_N1), in_dtype)
            D_tmem = T.alloc_tmem([block_M, block_N2], accum_dtype)
            mbar2 = T.alloc_barrier(1)

            D_local = T.alloc_fragment((block_M, block_N2), accum_dtype)
            D_shared = T.alloc_shared((block_M, block_N2), out_dtype)

            # Stage 1: SS variant — A and B1 both from shared memory
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B1[0, k * block_K], B1_shared)
                T.gemm(
                    A_shared, B1_shared, C_tmem,
                    transpose_A=False, transpose_B=True,
                    mbar=mbar1, wg_wait=-1, clear_accum=k == 0,
                )
                T.mbarrier_wait_parity(mbar1, k % 2)

            # Stage 2: TS variant — A (C_tmem) from TMEM, B2 from shared memory
            T.copy(B2[bx * block_N2, 0], B2_shared)
            T.gemm(
                C_tmem, B2_shared, D_tmem,
                transpose_A=False, transpose_B=True,
                mbar=mbar2, wg_wait=-1, clear_accum=True,
            )
            T.mbarrier_wait_parity(mbar2, 0)

            # Copy result out
            T.copy(D_tmem, D_local)
            T.copy(D_local, D_shared)
            T.copy(D_shared, D[by * block_M, bx * block_N2])

    return main


if __name__ == "__main__":
    M, N1, K, N2 = 128, 128, 128, 128
    block_M, block_N1, block_K, block_N2 = 128, 128, 128, 128
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    threads = 256

    func = chained_matmul(
        M, N1, K, N2,
        block_M, block_N1, block_K, block_N2,
        in_dtype, out_dtype, accum_dtype, threads,
    )

    jit_kernel = tilelang.compile(
        func,
        out_idx=[3],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    print(jit_kernel.get_kernel_source())

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b1 = torch.randn(N1, K, device="cuda", dtype=torch.bfloat16)
    b2 = torch.randn(N2, N1, device="cuda", dtype=torch.bfloat16)
    d = jit_kernel(a, b1, b2)

    ref_c = (a.float() @ b1.float().T)
    ref_d = (ref_c @ b2.float().T).to(torch.bfloat16)
    torch.testing.assert_close(d, ref_d, rtol=1e-1, atol=1e-1)
    print("Correctness check passed!")

    profiler = jit_kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"Latency: {latency} ms")
