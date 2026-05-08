# Introduce CLC tile schedule

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def get_swizzled_block_idx(tile_id, group_size, m_clusters, cta_id):
    bx_cluster = (tile_id // group_size) % m_clusters
    bx = bx_cluster * 2 + cta_id
    by = (tile_id % group_size) + (tile_id // group_size) // m_clusters * group_size
    return bx, by


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def gemm_clc_persistent_2cta(
    A,
    B,
    block_M,
    block_N,
    store_block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    group_size=8,
    use_tma_store=True,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
    C = T.empty((M, N), out_dtype)

    m_blocks = T.ceildiv(M, block_M)
    m_clusters = m_blocks // 2
    n_blocks = T.ceildiv(N, block_N)
    total_cluster_tiles = m_clusters * n_blocks
    k_blocks = T.ceildiv(K, block_K)
    assert n_blocks % (2 * group_size) == 0

    with T.Kernel(total_cluster_tiles * 2, threads=256, cluster_dims=2) as block_id:
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N // 2), in_dtype)
        C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)
        loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
        consumed = T.alloc_cluster_barrier([1] * num_stages)
        tmem_full = T.alloc_cluster_barrier([1] * 2)
        tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)
        schedule_arrived = T.alloc_cluster_barrier([1])
        schedule_finished = T.alloc_cluster_barrier([7])
        clc_result = T.alloc_shared((4,), "uint32", scope="shared")
        schedule_valid = T.alloc_shared((1,), "int32")
        schedule_tile_id = T.alloc_shared((1,), "int32")

        tx = T.get_thread_binding()
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)

        if tx < 32:  # Producer (TMA loads)
            for work_iter in T.unroll(total_cluster_tiles):
                if work_iter > 0:
                    T.mbarrier_wait_parity(schedule_arrived, (work_iter - 1) & 1)
                    if tx == 0:
                        T.mbarrier_arrive(schedule_finished, 0)
                    if schedule_valid[0] == 0:
                        break

                tile_id = T.if_then_else(
                    work_iter == 0,
                    block_id // 2,
                    schedule_tile_id[0],
                )
                bx, by = get_swizzled_block_idx(tile_id, group_size, m_clusters, cta_id)

                for k in T.serial(k_blocks):
                    phase = work_iter * k_blocks + k
                    T.mbarrier_wait_parity(consumed[phase % num_stages], ((phase // num_stages) & 1) ^ 1)
                    T.tma_copy(
                        A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                        A_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.tma_copy(
                        B[k * block_K : (k + 1) * block_K, (by * 2 + cta_id) * block_N // 2 : (by * 2 + cta_id + 1) * block_N // 2],
                        B_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.mbarrier_arrive(loaded[phase % num_stages], 0)

        elif cta_id == 0 and tx < 64:  # MMA (cta_id 0 only)
            for work_iter in T.unroll(total_cluster_tiles):
                if work_iter > 0:
                    T.mbarrier_wait_parity(schedule_arrived, (work_iter - 1) & 1)
                    if tx == 32:
                        T.mbarrier_arrive(schedule_finished, 0)
                    if schedule_valid[0] == 0:
                        break

                T.mbarrier_wait_parity(tmem_empty[work_iter & 1], ((work_iter // 2) & 1) ^ 1)
                for k in T.serial(k_blocks):
                    phase = work_iter * k_blocks + k
                    T.mbarrier_wait_parity(loaded[phase % num_stages], (phase // num_stages) & 1)
                    if work_iter & 1 == 0:
                        T.tcgen05_gemm(
                            A_shared[phase % num_stages, :, :],
                            B_shared[phase % num_stages, :, :],
                            C_tmem_0,
                            mbar=consumed[phase % num_stages],
                            clear_accum=k == 0,
                            use_2cta=True,
                        )
                    else:
                        T.tcgen05_gemm(
                            A_shared[phase % num_stages, :, :],
                            B_shared[phase % num_stages, :, :],
                            C_tmem_1,
                            mbar=consumed[phase % num_stages],
                            clear_accum=k == 0,
                            use_2cta=True,
                        )
                T.tcgen05_mma_arrive(tmem_full[work_iter & 1], arrive_2cta=True)

        elif 64 <= tx < 96:  # CLC Scheduler (both CTAs)
            for work_iter in T.unroll(total_cluster_tiles):
                if tx == 64:
                    if cta_id == 0 and work_iter > 0:
                        T.mbarrier_wait_parity(schedule_finished, (work_iter - 1) & 1)
                    T.mbarrier_arrive_expect_tx(schedule_arrived, 16)
                    if cta_id == 0:
                        T.clc_try_cancel_multicast(clc_result, schedule_arrived)
                    T.mbarrier_wait_parity(schedule_arrived, work_iter & 1)
                    schedule_valid[0] = T.clc_is_canceled(clc_result)
                    schedule_tile_id[0] = T.cast(T.clc_get_first_ctaid_x(clc_result), "int32") // 2
                    T.mbarrier_arrive(schedule_finished, 0)
                    if schedule_valid[0] == 0:
                        break

        elif 128 <= tx < 256:  # Epilogue
            for work_iter in T.unroll(total_cluster_tiles):
                if work_iter > 0:
                    T.mbarrier_wait_parity(schedule_arrived, (work_iter - 1) & 1)
                    if tx == 128:
                        T.mbarrier_arrive(schedule_finished, 0)
                    if schedule_valid[0] == 0:
                        break

                tile_id = T.if_then_else(
                    work_iter == 0,
                    block_id // 2,
                    schedule_tile_id[0],
                )
                bx, by = get_swizzled_block_idx(tile_id, group_size, m_clusters, cta_id)

                T.mbarrier_wait_parity(tmem_full[work_iter & 1], (work_iter // 2) & 1)
                T.sync_threads(1, 128)
                if work_iter & 1 == 0:
                    T.copy(C_tmem_0, C_local)
                else:
                    T.copy(C_tmem_1, C_local)
                T.mbarrier_arrive(tmem_empty[work_iter & 1], 0)

                if use_tma_store:
                    for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                        T.copy(C_local[:, i * store_block_N : (i + 1) * store_block_N], C_shared)
                        T.sync_threads(3, 128)
                        T.copy(C_shared, C[bx * block_M, by * block_N + i * store_block_N])
                        T.sync_threads(3, 128)
                else:
                    T.copy(C_local, C_local_cast)
                    T.copy(C_local_cast, C[bx * block_M, by * block_N])

    return C


def main():
    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K = 128, 256, 64
    store_block_N = 64
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    num_stages = 6
    l2_swizzle_group_size = 8

    kernel_args = (block_M, block_N, store_block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages, l2_swizzle_group_size)

    # a = (torch.rand(M, K, device="cuda", dtype=torch.bfloat16) * 2 - 1)
    # b = (torch.rand(K, N, device="cuda", dtype=torch.bfloat16) * 2 - 1)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    print(gemm_clc_persistent_2cta.get_kernel_source(a, b, *kernel_args))
    c = gemm_clc_persistent_2cta(a, b, *kernel_args)

    ref_c = (a.to(torch.float) @ b.to(torch.float)).to(torch.bfloat16)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All checks passed. ✅")

    tl_latency = do_bench(lambda: gemm_clc_persistent_2cta(a, b, *kernel_args), backend="cupti")
    torch_latency = do_bench(lambda: a @ b, backend="cupti")
    print(f"Tilelang latency: {tl_latency} ms")
    print(f"Flops: {2 * M * N * K / (tl_latency / 1e3) / 1e12} TFLOPS")
    print(f"Torch latency: {torch_latency} ms")
    print(f"Flops: {2 * M * N * K / (torch_latency / 1e3) / 1e12} TFLOPS")


if __name__ == "__main__":
    main()
