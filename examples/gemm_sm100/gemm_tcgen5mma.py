import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def matmul(
    A,
    B,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    M, N, K = T.const("M, N, K")
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    A: T.Tensor(A_shape, in_dtype)
    B: T.Tensor(B_shape, in_dtype)
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared(A_shared_shape, in_dtype)
        B_shared = T.alloc_shared(B_shared_shape, in_dtype)
        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
        mbar = T.alloc_barrier(1)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)  # not trans_A
            T.copy(B[bx * block_N, k * block_K], B_shared)  # trans_B
            T.tcgen05_gemm(A_shared, B_shared, C_tmem, trans_A, trans_B, mbar=mbar, clear_accum=k == 0)
            T.mbarrier_wait_parity(mbar, k % 2)

        T.copy(C_tmem, C_local)
        T.copy(C_local, C_shared)

        T.copy(C_shared, C[by * block_M, bx * block_N])

    return C


M, N, K = 4096, 4096, 8192
block_M, block_N, block_K = 128, 128, 128
trans_A, trans_B = False, True
in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
num_stages = 0 if block_N >= 256 or block_M >= 256 or block_K >= 256 else 2
threads = 256

a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
c = matmul(a, b, block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype, accum_dtype, num_stages, threads)
print(matmul.get_kernel_source(a, b, block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype, accum_dtype, num_stages, threads))

ref_c = (a.to(torch.float) @ b.T.to(torch.float)).to(torch.bfloat16)
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("All checks passed. ✅")

tl_latency = do_bench(
    lambda: matmul(a, b, block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype, accum_dtype, num_stages, threads),
    backend="cupti",
)
torch_latency = do_bench(lambda: a @ b.T, backend="cupti")
print(f"Tilelang latency: {tl_latency} ms")
print(f"Flops: {2 * M * N * K / (tl_latency / 1e3) / 1e12} TFLOPS")
print(f"Torch latency: {torch_latency} ms")
print(f"Flops: {2 * M * N * K / (torch_latency / 1e3) / 1e12} TFLOPS")
