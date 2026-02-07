# Persistent, 1-SM, num_epi_stages = 2

import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_LOOP_UNSWITCHING: True})
def gemm(
    A,
    B,
    block_M,
    block_N,
    store_block_N,  # block_N for C_shared
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    use_tma_store=True,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
    C = T.empty((M, N), out_dtype)

    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    assert K % (2 * block_K) == 0  # for simplicity
    k_blocks = T.ceildiv(K, block_K)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 8

    with T.Kernel(sm_num, threads=256) as (block_id):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N), in_dtype)
        C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)
        loaded = T.alloc_barrier([32] * num_stages)
        consumed = T.alloc_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1] * 2)
        tmem_empty = T.alloc_barrier([128] * 2)

        tx = T.get_thread_binding()

        if tx < 32:  # warp 0: issue tma
            for w in T.unroll(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    for k in T.serial(k_blocks):
                        T.mbarrier_wait_parity(consumed[k % num_stages], ((k // num_stages) & 1) ^ 1)
                        T.copy(
                            A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared[k % num_stages, :, :]
                        )  # cannot use BufferLoad here
                        T.copy(B[k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_shared[k % num_stages, :, :])
                        T.mbarrier_arrive(loaded[k % num_stages])

        elif tx < 64:  # warp 1: issue tcgen5
            for w in T.unroll(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.mbarrier_wait_parity(tmem_empty[w & 1], ((w // 2) & 1) ^ 1)
                    for k in T.serial(k_blocks):
                        T.mbarrier_wait_parity(loaded[k % num_stages], (k // num_stages) & 1)
                        if w & 1 == 0:
                            T.gemm(
                                A_shared[k % num_stages, :, :],
                                B_shared[k % num_stages, :, :],
                                C_tmem_0,
                                False,
                                False,
                                mbar=consumed[k % num_stages],
                                wg_wait=-1,
                                clear_accum=k == 0,
                            )
                        else:
                            T.gemm(
                                A_shared[k % num_stages, :, :],
                                B_shared[k % num_stages, :, :],
                                C_tmem_1,
                                False,
                                False,
                                mbar=consumed[k % num_stages],
                                wg_wait=-1,
                                clear_accum=k == 0,
                            )
                    T.tcgen05_mma_arrive(tmem_full[w & 1])

        elif 128 <= tx < 256:  # warp 4~7: epilogue
            for w in T.unroll(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.mbarrier_wait_parity(tmem_full[w & 1], (w // 2) & 1)
                    T.sync_threads(1, 128)
                    if (w & 1) == 0:
                        T.copy(C_tmem_0, C_local)
                    else:
                        T.copy(C_tmem_1, C_local)
                    T.mbarrier_arrive(tmem_empty[w & 1])

                    if use_tma_store:
                        for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                            T.copy(C_local[:, i * store_block_N : (i + 1) * store_block_N], C_shared)
                            T.copy(C_shared, C[bx * block_M, by * block_N + i * store_block_N])
                    else:
                        T.copy(C_local, C_local_cast)
                        T.copy(C_local_cast, C[bx * block_M, by * block_N])
    return C


def main():
    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K = 128, 256, 64
    store_block_N = 128
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    num_stages = 4

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    print(gemm.get_kernel_source(a, b, block_M, block_N, store_block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages))
    c = gemm(a, b, block_M, block_N, store_block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages)

    ref_c = (a.to(torch.float) @ b.to(torch.float)).to(torch.bfloat16)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All checks passed. ✅")

    tl_latency = do_bench(
        lambda: gemm(a, b, block_M, block_N, store_block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages), backend="cupti"
    )
    torch_latency = do_bench(lambda: a @ b, backend="cupti")
    print(f"Tilelang latency: {tl_latency} ms")
    print(f"Flops: {2 * M * N * K / (tl_latency / 1e3) / 1e12} TFLOPS")
    print(f"Torch latency: {torch_latency} ms")
    print(f"Flops: {2 * M * N * K / (torch_latency / 1e3) / 1e12} TFLOPS")


if __name__ == "__main__":
    main()
