# Persistent, warp-specialized TCGEN05 GEMMs.
#
# ``gemm_persistent`` and ``gemm_persistent_2cta`` use the static
# PersistentTileScheduler. ``gemm_streamk_2cta`` splits the first wave along K
# and uses a workspace fixup before computing the remaining data-parallel
# tiles.

import argparse

import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


def get_streamk_tiles(total_tiles, num_clusters):
    """Split only the under-filled tail wave across the resident clusters."""
    return total_tiles % num_clusters


@tilelang.jit
def gemm_persistent(
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
    group_size,
    use_tma_store,
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
    assert n_blocks % (2 * group_size) == 0  # Please adjust group_size if not satisfied

    with T.Kernel(sm_num, threads=256) as (block_id):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N), in_dtype)
        C_tmem = T.alloc_tmem([2, block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
        if use_tma_store:
            C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)
        else:
            C_local_cast = T.alloc_fragment((block_M, store_block_N), out_dtype)
        loaded = T.alloc_barrier([32] * num_stages)
        consumed = T.alloc_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1] * 2)
        tmem_empty = T.alloc_barrier([128] * 2)

        tx = T.get_thread_binding()

        if tx < 32:  # warp 0: issue tma
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size)
            sched.init(block_id)
            while sched.valid():
                bx, by = sched.m_idx, sched.n_idx

                for k in T.serial(k_blocks):
                    phase = sched.current_iter * k_blocks + k
                    T.mbarrier_wait_parity(consumed[phase % num_stages], ((phase // num_stages) & 1) ^ 1)
                    T.tma_copy(
                        A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                        A_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.tma_copy(
                        B[k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N],
                        B_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.mbarrier_arrive(loaded[phase % num_stages])
                sched.next_tile()

        elif tx < 64:  # warp 1: issue tcgen5
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size)
            sched.init(block_id)
            while sched.valid():
                T.mbarrier_wait_parity(tmem_empty[sched.current_iter & 1], ((sched.current_iter // 2) & 1) ^ 1)
                for k in T.serial(k_blocks):
                    phase = sched.current_iter * k_blocks + k
                    T.mbarrier_wait_parity(loaded[phase % num_stages], (phase // num_stages) & 1)
                    T.tcgen05_gemm(
                        A_shared[phase % num_stages, :, :],
                        B_shared[phase % num_stages, :, :],
                        C_tmem[sched.current_iter & 1, :, :],
                        mbar=consumed[phase % num_stages],
                        clear_accum=k == 0,
                    )
                T.tcgen05_mma_arrive(tmem_full[sched.current_iter & 1])
                sched.next_tile()

        elif 128 <= tx < 256:  # warp 4~7: epilogue
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size)
            sched.init(block_id)
            while sched.valid():
                bx, by = sched.m_idx, sched.n_idx

                T.mbarrier_wait_parity(tmem_full[sched.current_iter & 1], (sched.current_iter // 2) & 1)

                for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                    T.copy(
                        C_tmem[
                            sched.current_iter & 1,
                            :,
                            i * store_block_N : (i + 1) * store_block_N,
                        ],
                        C_local,
                    )
                    if use_tma_store:
                        T.copy(C_local, C_shared)
                        T.copy(C_shared, C[bx * block_M, by * block_N + i * store_block_N])
                    else:
                        T.copy(C_local, C_local_cast)
                        T.copy(
                            C_local_cast,
                            C[bx * block_M, by * block_N + i * store_block_N],
                        )

                T.mbarrier_arrive(tmem_empty[sched.current_iter & 1])

                sched.next_tile()
    return C


@tilelang.jit
def gemm_persistent_2cta(
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
    group_size,
    use_tma_store,
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
    assert n_blocks % (2 * group_size) == 0  # Please adjust group_size if not satisfied
    cluster_size = 2

    with T.ClusterKernel(sm_num, threads=256, cluster_dims=2) as (block_id):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N // 2), in_dtype)
        C_tmem = T.alloc_tmem([2, block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
        if use_tma_store:
            C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)
        else:
            C_local_cast = T.alloc_fragment((block_M, store_block_N), out_dtype)
        loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
        consumed = T.alloc_cluster_barrier([1] * num_stages)
        tmem_full = T.alloc_cluster_barrier([1] * 2)
        tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)

        tx = T.get_thread_binding()
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)  # todo: automatically assume this

        if tx < 32:  # warp 0: issue tma
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size, cluster_size=cluster_size)
            sched.init(block_id // cluster_size)
            while sched.valid():
                bx, by = sched.m_idx * cluster_size + cta_id, sched.n_idx

                for k in T.serial(k_blocks):
                    phase = sched.current_iter * k_blocks + k
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
                sched.next_tile()

        elif tx < 64 and cta_id == 0:  # warp 1: issue tcgen5
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size, cluster_size=cluster_size)
            sched.init(block_id // cluster_size)
            while sched.valid():
                T.mbarrier_wait_parity(tmem_empty[sched.current_iter & 1], ((sched.current_iter // 2) & 1) ^ 1)
                for k in T.serial(k_blocks):
                    phase = sched.current_iter * k_blocks + k
                    T.mbarrier_wait_parity(loaded[phase % num_stages], (phase // num_stages) & 1)
                    T.tcgen05_gemm(
                        A_shared[phase % num_stages, :, :],
                        B_shared[phase % num_stages, :, :],
                        C_tmem[sched.current_iter & 1, :, :],
                        mbar=consumed[phase % num_stages],
                        clear_accum=k == 0,
                        use_2cta=True,
                    )
                T.tcgen05_mma_arrive(tmem_full[sched.current_iter & 1], arrive_2cta=True)
                sched.next_tile()

        elif 128 <= tx < 256:  # warp 4~7: epilogue
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size, cluster_size=cluster_size)
            sched.init(block_id // cluster_size)
            while sched.valid():
                bx, by = sched.m_idx * cluster_size + cta_id, sched.n_idx

                T.mbarrier_wait_parity(tmem_full[sched.current_iter & 1], (sched.current_iter // 2) & 1)

                for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                    T.copy(
                        C_tmem[
                            sched.current_iter & 1,
                            :,
                            i * store_block_N : (i + 1) * store_block_N,
                        ],
                        C_local,
                    )
                    if use_tma_store:
                        T.copy(C_local, C_shared)
                        T.copy(C_shared, C[bx * block_M, by * block_N + i * store_block_N])
                    else:
                        T.copy(C_local, C_local_cast)
                        T.copy(
                            C_local_cast,
                            C[bx * block_M, by * block_N + i * store_block_N],
                        )

                T.mbarrier_arrive(tmem_empty[sched.current_iter & 1], 0)

                sched.next_tile()

    return C


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def gemm_streamk_2cta(
    A,
    B,
    Workspace,
    PartialTile,
    Fixup,
    streamk_tiles,
    block_M,
    block_N,
    store_block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    use_tma_store=True,
):
    """Persistent 2-CTA TCGEN05 GEMM with Stream-K decomposition.

    Each resident cluster receives a contiguous interval of K iterations from
    the first wave. Complete output tiles bypass the workspace. Split tiles use
    CUTLASS-style separate reduction: peers publish independent partials in
    parallel, then the peer computing the final K interval performs the fixup
    and epilogue. Remaining tiles use ordinary data-parallel persistent
    scheduling.
    """
    M, N, K = T.const("M, N, K")
    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]

    sm_num = driver.get_num_sms()
    num_clusters = sm_num // 2
    Workspace: T.Tensor[[num_clusters, 2, block_M, block_N], accum_dtype]
    # Processing each interval backwards means that only its trailing partial
    # needs to be published; a leading partial performs the tile's fixup.
    # Initialize this to -1 once; its deterministic entries are overwritten on
    # every launch and can subsequently be reused.
    PartialTile: T.Tensor[[num_clusters], T.int32]
    # Fixup must be zero-initialized once. The final peer resets its tile lock,
    # so the workspace can be reused by subsequent launches on the same stream.
    Fixup: T.Tensor[[streamk_tiles], T.int32]
    C = T.empty((M, N), out_dtype)

    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    m_clusters = m_blocks // 2
    total_tiles = m_clusters * n_blocks
    k_blocks = T.ceildiv(K, block_K)
    blocking_tiles = total_tiles - streamk_tiles
    streamk_iters = streamk_tiles * k_blocks
    streamk_full_iters = streamk_iters // num_clusters
    streamk_partial_iters = streamk_iters % num_clusters
    blocking_waves = blocking_tiles // num_clusters

    assert sm_num % 2 == 0
    assert M % (2 * block_M) == 0
    assert N % block_N == 0
    assert K % (2 * block_K) == 0
    assert 0 < streamk_tiles <= total_tiles
    assert blocking_tiles % num_clusters == 0

    with T.ClusterKernel(sm_num, threads=256, cluster_dims=2) as block_id:
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N // 2), in_dtype)
        C_tmem = T.alloc_tmem([2, block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
        C_partial = T.alloc_fragment((block_M, store_block_N), accum_dtype)
        C_local_cast = T.alloc_fragment((block_M, store_block_N), out_dtype)
        if use_tma_store:
            C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)

        loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
        consumed = T.alloc_cluster_barrier([1] * num_stages)
        tmem_full = T.alloc_cluster_barrier([1] * 2)
        tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)
        fixup_stored = T.alloc_cluster_barrier([128 * 2] * 2)
        fixup_finished = T.alloc_cluster_barrier([128 * 2] * 2)
        fixup_ready = T.alloc_barrier([1] * 2)

        tx = T.get_thread_binding()
        cta_id = T.block_rank_in_cluster()
        cluster_id = block_id // 2
        T.assume(cta_id < 2)

        if num_clusters % streamk_tiles == 0:
            # Keep split boundaries within an output tile when resident
            # clusters can be divided evenly among the Stream-K tiles. This
            # avoids publishing a full accumulator for a tiny cross-tile
            # interval at a work-unit boundary.
            clusters_per_tile = num_clusters // streamk_tiles
            streamk_tile = cluster_id // clusters_per_tile
            split_id = cluster_id % clusters_per_tile
            split_full_iters = k_blocks // clusters_per_tile
            split_partial_iters = k_blocks % clusters_per_tile
            start_iter = streamk_tile * k_blocks + split_id * split_full_iters + T.min(split_id, split_partial_iters)
            last_iter = streamk_tile * k_blocks + (split_id + 1) * split_full_iters + T.min(split_id + 1, split_partial_iters)
        else:
            start_iter = cluster_id * streamk_full_iters + T.min(cluster_id, streamk_partial_iters)
            last_iter = (cluster_id + 1) * streamk_full_iters + T.min(cluster_id + 1, streamk_partial_iters)

        if tx < 32:  # warp 0: issue TMA
            current_iter = T.alloc_var(T.int32, init=last_iter)
            phase = T.alloc_var(T.int32, init=0)
            while current_iter > start_iter:
                tile_id = (current_iter - 1) // k_blocks
                begin_iter = T.max(tile_id * k_blocks, start_iter)
                bx = (tile_id // n_blocks) * 2 + cta_id
                by = tile_id % n_blocks
                k_start = begin_iter % k_blocks

                for k in T.serial(current_iter - begin_iter):
                    k_idx = k_start + k
                    T.mbarrier_wait_parity(
                        consumed[phase % num_stages],
                        ((phase // num_stages) & 1) ^ 1,
                    )
                    T.tma_copy(
                        A[bx * block_M : (bx + 1) * block_M, k_idx * block_K : (k_idx + 1) * block_K],
                        A_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.tma_copy(
                        B[
                            k_idx * block_K : (k_idx + 1) * block_K,
                            (by * 2 + cta_id) * block_N // 2 : (by * 2 + cta_id + 1) * block_N // 2,
                        ],
                        B_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.mbarrier_arrive(loaded[phase % num_stages], 0)
                    phase += 1
                current_iter = begin_iter

            for wave in T.serial(blocking_waves):
                tile_id = streamk_tiles + cluster_id + wave * num_clusters
                bx = (tile_id // n_blocks) * 2 + cta_id
                by = tile_id % n_blocks
                for k in T.serial(k_blocks):
                    T.mbarrier_wait_parity(
                        consumed[phase % num_stages],
                        ((phase // num_stages) & 1) ^ 1,
                    )
                    T.tma_copy(
                        A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                        A_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.tma_copy(
                        B[
                            k * block_K : (k + 1) * block_K,
                            (by * 2 + cta_id) * block_N // 2 : (by * 2 + cta_id + 1) * block_N // 2,
                        ],
                        B_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.mbarrier_arrive(loaded[phase % num_stages], 0)
                    phase += 1

        elif tx < 64 and cta_id == 0:  # warp 1: issue TCGEN05 MMA
            current_iter = T.alloc_var(T.int32, init=last_iter)
            phase = T.alloc_var(T.int32, init=0)
            work_iter = T.alloc_var(T.int32, init=0)
            while current_iter > start_iter:
                tile_id = (current_iter - 1) // k_blocks
                begin_iter = T.max(tile_id * k_blocks, start_iter)
                T.mbarrier_wait_parity(
                    tmem_empty[work_iter & 1],
                    ((work_iter // 2) & 1) ^ 1,
                )
                for k in T.serial(current_iter - begin_iter):
                    T.mbarrier_wait_parity(
                        loaded[phase % num_stages],
                        (phase // num_stages) & 1,
                    )
                    T.tcgen05_gemm(
                        A_shared[phase % num_stages, :, :],
                        B_shared[phase % num_stages, :, :],
                        C_tmem[work_iter & 1, :, :],
                        mbar=consumed[phase % num_stages],
                        clear_accum=k == 0,
                        use_2cta=True,
                    )
                    phase += 1
                T.tcgen05_mma_arrive(tmem_full[work_iter & 1], arrive_2cta=True)
                current_iter = begin_iter
                work_iter += 1

            for _wave in T.serial(blocking_waves):
                T.mbarrier_wait_parity(
                    tmem_empty[work_iter & 1],
                    ((work_iter // 2) & 1) ^ 1,
                )
                for k in T.serial(k_blocks):
                    T.mbarrier_wait_parity(
                        loaded[phase % num_stages],
                        (phase // num_stages) & 1,
                    )
                    T.tcgen05_gemm(
                        A_shared[phase % num_stages, :, :],
                        B_shared[phase % num_stages, :, :],
                        C_tmem[work_iter & 1, :, :],
                        mbar=consumed[phase % num_stages],
                        clear_accum=k == 0,
                        use_2cta=True,
                    )
                    phase += 1
                T.tcgen05_mma_arrive(tmem_full[work_iter & 1], arrive_2cta=True)
                work_iter += 1

        elif 128 <= tx < 256:  # warp 4~7: epilogue and Stream-K fixup
            current_iter = T.alloc_var(T.int32, init=last_iter)
            work_iter = T.alloc_var(T.int32, init=0)
            store_iter = T.alloc_var(T.int32, init=0)
            final_iter = T.alloc_var(T.int32, init=0)
            while current_iter > start_iter:
                tile_id = (current_iter - 1) // k_blocks
                begin_iter = T.max(tile_id * k_blocks, start_iter)
                bx = (tile_id // n_blocks) * 2 + cta_id
                by = tile_id % n_blocks
                k_start = begin_iter % k_blocks
                k_count = current_iter - begin_iter
                is_full_tile = k_start == 0 and k_count == k_blocks
                computes_epilogue = current_iter % k_blocks == 0

                T.mbarrier_wait_parity(
                    tmem_full[work_iter & 1],
                    (work_iter // 2) & 1,
                )

                for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                    T.copy(
                        C_tmem[
                            work_iter & 1,
                            :,
                            i * store_block_N : (i + 1) * store_block_N,
                        ],
                        C_local,
                    )
                    if is_full_tile:
                        T.copy(C_local, C_local_cast)
                        T.copy(
                            C_local_cast,
                            C[bx * block_M, by * block_N + i * store_block_N],
                        )
                    elif not computes_epilogue:
                        T.copy(
                            C_local,
                            Workspace[
                                cluster_id,
                                cta_id,
                                :,
                                i * store_block_N : (i + 1) * store_block_N,
                            ],
                        )

                if not is_full_tile and not computes_epilogue:
                    T.mbarrier_arrive(fixup_stored[store_iter & 1], 0)
                    if tx == 128 and cta_id == 0:
                        T.mbarrier_wait_parity(
                            fixup_stored[store_iter & 1],
                            (store_iter // 2) & 1,
                        )
                        if num_clusters % streamk_tiles != 0:
                            PartialTile[cluster_id] = tile_id
                        T.atomic_add(Fixup[tile_id], k_count, memory_order="release")
                    store_iter += 1

                if not is_full_tile and computes_epilogue:
                    if tx == 128:
                        fixup_value = T.alloc_var(
                            T.int32,
                            init=T.atomic_load(Fixup[tile_id], memory_order="acquire"),
                        )
                        while fixup_value < k_start:
                            fixup_value = T.atomic_load(Fixup[tile_id], memory_order="acquire")
                        T.mbarrier_arrive(fixup_ready[final_iter & 1])
                    T.mbarrier_wait_parity(
                        fixup_ready[final_iter & 1],
                        (final_iter // 2) & 1,
                    )

                    for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                        T.copy(
                            C_tmem[
                                work_iter & 1,
                                :,
                                i * store_block_N : (i + 1) * store_block_N,
                            ],
                            C_local,
                        )
                        if num_clusters % streamk_tiles == 0:
                            first_producer = tile_id * clusters_per_tile
                            for peer in T.serial(clusters_per_tile - 1):
                                producer = first_producer + peer
                                T.copy(
                                    Workspace[
                                        producer,
                                        cta_id,
                                        :,
                                        i * store_block_N : (i + 1) * store_block_N,
                                    ],
                                    C_partial,
                                )
                                for x, y in T.Parallel(block_M, store_block_N):
                                    C_local[x, y] += C_partial[x, y]
                        else:
                            for producer in T.serial(num_clusters):
                                if PartialTile[producer] == tile_id:
                                    T.copy(
                                        Workspace[
                                            producer,
                                            cta_id,
                                            :,
                                            i * store_block_N : (i + 1) * store_block_N,
                                        ],
                                        C_partial,
                                    )
                                    for x, y in T.Parallel(block_M, store_block_N):
                                        C_local[x, y] += C_partial[x, y]
                        T.copy(C_local, C_local_cast)
                        T.copy(
                            C_local_cast,
                            C[bx * block_M, by * block_N + i * store_block_N],
                        )

                    T.mbarrier_arrive(fixup_finished[final_iter & 1], 0)
                    if tx == 128 and cta_id == 0:
                        T.mbarrier_wait_parity(
                            fixup_finished[final_iter & 1],
                            (final_iter // 2) & 1,
                        )
                        T.atomic_store(Fixup[tile_id], 0, memory_order="release")
                    final_iter += 1
                T.mbarrier_arrive(tmem_empty[work_iter & 1], 0)
                current_iter = begin_iter
                work_iter += 1

            for wave in T.serial(blocking_waves):
                tile_id = streamk_tiles + cluster_id + wave * num_clusters
                bx = (tile_id // n_blocks) * 2 + cta_id
                by = tile_id % n_blocks
                T.mbarrier_wait_parity(
                    tmem_full[work_iter & 1],
                    (work_iter // 2) & 1,
                )
                for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                    T.copy(
                        C_tmem[
                            work_iter & 1,
                            :,
                            i * store_block_N : (i + 1) * store_block_N,
                        ],
                        C_local,
                    )
                    if use_tma_store:
                        T.copy(C_local, C_shared)
                        T.sync_threads(3, 128)
                        T.copy(C_shared, C[bx * block_M, by * block_N + i * store_block_N])
                        T.sync_threads(3, 128)
                    else:
                        T.copy(C_local, C_local_cast)
                        T.copy(
                            C_local_cast,
                            C[bx * block_M, by * block_N + i * store_block_N],
                        )
                T.mbarrier_arrive(tmem_empty[work_iter & 1], 0)
                work_iter += 1

    return C


def main():
    parser = argparse.ArgumentParser(description="Persistent warp-specialized TCGEN05 GEMM")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=256)
    parser.add_argument("--block_K", type=int, default=64)
    parser.add_argument("--store_block_N", type=int, default=64, help="block_N for C_shared")
    parser.add_argument("--num_stages", type=int, default=None, help="pipeline stages (default: 6 with 2cta, else 4)")
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument(
        "--scheduler",
        choices=("persistent", "streamk"),
        default="persistent",
    )
    parser.add_argument("--enable_2cta_tcgen5mma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_tma_store", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    enable_2cta_tcgen5mma = args.enable_2cta_tcgen5mma
    if args.num_stages is not None:
        num_stages = args.num_stages
    else:
        num_stages = 6 if enable_2cta_tcgen5mma else 4  # Each cta only needs to load half of B, enabling larger stages
    if args.scheduler == "streamk" and not enable_2cta_tcgen5mma:
        parser.error("the Stream-K example currently requires 2-CTA TCGEN05 MMA")

    kernel = gemm_persistent_2cta if enable_2cta_tcgen5mma else gemm_persistent
    kwargs = {
        "block_M": args.block_M,
        "block_N": args.block_N,
        "store_block_N": args.store_block_N,
        "block_K": args.block_K,
        "in_dtype": in_dtype,
        "out_dtype": out_dtype,
        "accum_dtype": accum_dtype,
        "num_stages": num_stages,
        "group_size": args.group_size,
        "use_tma_store": args.use_tma_store,
    }

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    kernel_args = (a, b)
    if args.scheduler == "streamk":
        num_clusters = driver.get_num_sms() // 2
        total_tiles = (M // (2 * args.block_M)) * (N // args.block_N)
        streamk_tiles = get_streamk_tiles(total_tiles, num_clusters)
        if streamk_tiles == 0:
            parser.error("this shape has no under-filled cluster wave; use the persistent scheduler")
        workspace = torch.empty(
            (num_clusters, 2, args.block_M, args.block_N),
            device="cuda",
            dtype=torch.float32,
        )
        partial_tile = torch.full(
            (num_clusters,),
            -1,
            device="cuda",
            dtype=torch.int32,
        )
        fixup = torch.zeros((streamk_tiles,), device="cuda", dtype=torch.int32)
        kernel = gemm_streamk_2cta
        kernel_args = (a, b, workspace, partial_tile, fixup)
        kwargs = {
            **kwargs,
            "streamk_tiles": streamk_tiles,
        }
        kwargs.pop("group_size")

    print(kernel.get_kernel_source(*kernel_args, **kwargs))
    c = kernel(*kernel_args, **kwargs)

    ref_c = (a.to(torch.float) @ b.to(torch.float)).to(torch.bfloat16)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All checks passed. ✅")

    tl_latency = do_bench(
        lambda: kernel(*kernel_args, **kwargs),
        _n_warmup=50,
        _n_repeat=50,
        backend="cupti",
    )
    torch_latency = do_bench(lambda: a @ b, backend="cupti")
    print(f"Tilelang latency: {tl_latency} ms")
    print(f"Flops: {2 * M * N * K / (tl_latency / 1e3) / 1e12} TFLOPS")
    print(f"Torch latency: {torch_latency} ms")
    print(f"Flops: {2 * M * N * K / (torch_latency / 1e3) / 1e12} TFLOPS")


if __name__ == "__main__":
    main()
