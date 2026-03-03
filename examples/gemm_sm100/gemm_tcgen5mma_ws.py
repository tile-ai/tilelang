# Non-persistent, 1-SM GEMM

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit
def gemm(A, B, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages, use_tma_store=True):
    M, N, K = T.const("M, N, K")

    k_iters = T.ceildiv(K, block_K)

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N), in_dtype)
        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        loaded = T.alloc_barrier([32] * num_stages)
        consumed = T.alloc_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1])

        tx = T.get_thread_binding()

        T.use_swizzle(8)

        if tx < 32:  # warp 0: issue tma
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(consumed[k % num_stages], ((k // num_stages) & 1) ^ 1)
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared[k % num_stages, :, :])
                T.copy(B[k * block_K : (k + 1) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[k % num_stages, :, :])
                T.mbarrier_arrive(loaded[k % num_stages])
        elif tx < 64:  # warp 1: issue tcgen5
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(loaded[k % num_stages], (k // num_stages) & 1)
                T.gemm(
                    A_shared[k % num_stages, :, :],
                    B_shared[k % num_stages, :, :],
                    C_tmem,
                    mbar=consumed[k % num_stages],
                    wg_wait=-1,
                    clear_accum=k == 0,
                )
            T.tcgen05_mma_arrive(tmem_full)

        # Wait for all tcgen5 to finish
        T.mbarrier_wait_parity(tmem_full, 0)

        T.sync_threads()  # TileLang won't generate this if not annotated
        T.copy(C_tmem, C_local)
        if use_tma_store:
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])
        else:
            T.copy(C_local, C_local_cast)
            T.copy(C_local_cast, C[by * block_M, bx * block_N])
    return C


def main():
    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K = 128, 256, 64
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    num_stages = 4

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = gemm(a, b, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages)
    print(gemm.get_kernel_source(a, b, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages))

    ref_c = (a.to(torch.float) @ b.to(torch.float)).to(torch.bfloat16)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All checks passed. ✅")

    tl_latency = do_bench(lambda: gemm(a, b, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages), backend="cupti")
    torch_latency = do_bench(lambda: a @ b, backend="cupti")
    print(f"Tilelang latency: {tl_latency} ms")
    print(f"Flops: {2 * M * N * K / (tl_latency / 1e3) / 1e12} TFLOPS")
    print(f"Torch latency: {torch_latency} ms")
    print(f"Flops: {2 * M * N * K / (torch_latency / 1e3) / 1e12} TFLOPS")


if __name__ == "__main__":
    main()
