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

        if tx < 32:
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

        elif cta_id == 0 and tx < 64:
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

        elif 64 <= tx < 96:
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

        elif 128 <= tx < 256:
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


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def gemm_clc_persistent_2cta_pipelined_clc(
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
    clc_stages=2,
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

        schedule_arrived = T.alloc_cluster_barrier([1] * clc_stages)
        schedule_finished = T.alloc_cluster_barrier([5] * clc_stages)
        clc_result = T.alloc_shared((clc_stages, 4), "uint32", scope="shared")
        schedule_valid = T.alloc_shared((clc_stages,), "int32")
        schedule_tile_id = T.alloc_shared((clc_stages,), "int32")

        tx = T.get_thread_binding()
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)

        if tx < 32:
            for work_iter in T.unroll(total_cluster_tiles):
                s_cons = (work_iter - 1) % clc_stages
                c_cons = (work_iter - 1) // clc_stages

                if work_iter > 0:
                    T.mbarrier_wait_parity(schedule_arrived[s_cons], c_cons & 1)
                    if tx == 0:
                        T.mbarrier_arrive(schedule_finished[s_cons], 0)
                    if schedule_valid[s_cons] == 0:
                        break

                tile_id = T.if_then_else(
                    work_iter == 0,
                    block_id // 2,
                    schedule_tile_id[s_cons],
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

        elif cta_id == 0 and tx < 64:
            for work_iter in T.unroll(total_cluster_tiles):
                s_cons = (work_iter - 1) % clc_stages
                c_cons = (work_iter - 1) // clc_stages

                if work_iter > 0:
                    T.mbarrier_wait_parity(schedule_arrived[s_cons], c_cons & 1)
                    if tx == 32:
                        T.mbarrier_arrive(schedule_finished[s_cons], 0)
                    if schedule_valid[s_cons] == 0:
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

        elif 64 <= tx < 96:
            for work_iter in T.unroll(total_cluster_tiles):
                s_clc = work_iter % clc_stages
                c_clc = work_iter // clc_stages

                if tx == 64:
                    if cta_id == 0 and c_clc > 0:
                        T.mbarrier_wait_parity(schedule_finished[s_clc], (c_clc - 1) & 1)

                    T.mbarrier_arrive_expect_tx(schedule_arrived[s_clc], 16)
                    if cta_id == 0:
                        T.clc_try_cancel_multicast(clc_result[s_clc, :], schedule_arrived[s_clc])
                    T.mbarrier_wait_parity(schedule_arrived[s_clc], c_clc & 1)
                    schedule_valid[s_clc] = T.clc_is_canceled(clc_result[s_clc, :])
                    schedule_tile_id[s_clc] = T.cast(T.clc_get_first_ctaid_x(clc_result[s_clc, :]), "int32") // 2
                    if schedule_valid[s_clc] == 0:
                        break

        elif 128 <= tx < 256:
            for work_iter in T.unroll(total_cluster_tiles):
                s_cons = (work_iter - 1) % clc_stages
                c_cons = (work_iter - 1) // clc_stages

                if work_iter > 0:
                    T.mbarrier_wait_parity(schedule_arrived[s_cons], c_cons & 1)
                    if tx == 128:
                        T.mbarrier_arrive(schedule_finished[s_cons], 0)
                    if schedule_valid[s_cons] == 0:
                        break

                tile_id = T.if_then_else(
                    work_iter == 0,
                    block_id // 2,
                    schedule_tile_id[s_cons],
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


def ref_program(A, B):
    return (A.to(torch.float) @ B.to(torch.float)).to(torch.bfloat16)


if __name__ == "__main__":
    block_M, block_N, store_block_N, block_K = 128, 256, 64, 64
    num_stages, group_size = 6, 8
    base_args = (block_M, block_N, store_block_N, block_K, T.bfloat16, T.bfloat16, T.float, num_stages)

    for M, N, K in ((4096, 4096, 4096), (8192, 8192, 8192), (16384, 16384, 16384)):
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        ref = ref_program(a, b)
        flops = 2 * M * N * K

        c = gemm_clc_persistent_2cta(a, b, *base_args, group_size)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
        ms_base = do_bench(lambda a=a, b=b: gemm_clc_persistent_2cta(a, b, *base_args, group_size), backend="event")

        ms_torch = do_bench(lambda a=a, b=b: a @ b, backend="event")
        print(f"M=N=K={M:<6}  base: {flops / (ms_base / 1e3) / 1e12:6.1f} TFLOPS  torch: {flops / (ms_torch / 1e3) / 1e12:6.1f} TFLOPS")

        for clc in (2, 3, 4):
            c2 = gemm_clc_persistent_2cta_pipelined_clc(a, b, *base_args, clc, group_size)
            torch.testing.assert_close(c2, ref, rtol=1e-2, atol=1e-2)
            ms_pipe = do_bench(
                lambda a=a, b=b, clc=clc: gemm_clc_persistent_2cta_pipelined_clc(a, b, *base_args, clc, group_size), backend="event"
            )
            print(f"             pipe(clc={clc}): {flops / (ms_pipe / 1e3) / 1e12:6.1f} TFLOPS  ({ms_base / ms_pipe:.3f}x vs base)")
