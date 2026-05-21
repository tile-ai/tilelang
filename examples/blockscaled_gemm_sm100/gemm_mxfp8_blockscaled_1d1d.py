# MXFP8 Block-Scaled GEMM on SM100
# Blockscale size: (M, N, K) = (1, 1, 128)

import argparse
import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


@tilelang.jit
def mxfp8_blockscaled_gemm(
    A,
    B,
    SFA,
    SFB,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    sf_granularity_k=128,
    transpose_B=False,
):
    """1D-1D Block-scaled MXFP8 GEMM.

    A:   [M, K] in FP8 (E4M3 or E5M2)
    B:   [K, N] in FP8 (E4M3 or E5M2), or [N, K] when transpose_B=True
    SFA: [(K / sf_granularity_k) / 4) * M] in uint32
         Group-major packed E8M0 scale factors for A.
    SFB: [(K / sf_granularity_k) / 4) * N] in uint32
         Group-major packed E8M0 scale factors for B.
    """
    M, N, K = T.const("M, N, K")

    k_iters = T.ceildiv(K, block_K)
    # Load 4 K-blocks of SF at once → load every 4 iterations
    sf_load_period = sf_granularity_k * 4 // block_K
    sf_k_groups = T.ceildiv(T.ceildiv(K, sf_granularity_k), 4)

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[N, K] if transpose_B else [K, N], in_dtype]
    SFA: T.Tensor[[sf_k_groups * M], T.uint32]
    SFB: T.Tensor[[sf_k_groups * N], T.uint32]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
        # Data shared memory (pipelined)
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared(
            (num_stages, block_N, block_K) if transpose_B else (num_stages, block_K, block_N),
            in_dtype,
        )

        # Scale factor shared memory — one uint32 per row/column, packing 4 K-blocks.
        SFA_shared = T.alloc_shared((num_stages, block_M), "uint32")
        SFB_shared = T.alloc_shared((num_stages, block_N), "uint32")

        # Accumulator in tensor memory
        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)

        # Scale factors in tensor memory (TMEM has 128 rows / 32-bit cells)
        SFA_tmem = T.alloc_tmem([block_M, block_M // 128 * 4], "uint32")
        SFB_tmem = T.alloc_tmem([block_M, block_N // 128 * 4], "uint32")

        # Output buffers
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        # Barriers
        loaded = T.alloc_barrier([32] * num_stages)
        with_sf_full = T.alloc_barrier([32] * num_stages)
        consumed = T.alloc_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1])

        tx = T.get_thread_binding()
        T.use_swizzle(8)

        if tx < 32:
            # Warp 0: TMA load
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(consumed[k % num_stages], ((k // num_stages) & 1) ^ 1)
                T.tma_copy(
                    A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                    A_shared[k % num_stages, :, :],
                    barrier=loaded[k % num_stages],
                )
                if transpose_B:
                    T.tma_copy(
                        B[by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K],
                        B_shared[k % num_stages, :, :],
                        barrier=loaded[k % num_stages],
                    )
                else:
                    T.tma_copy(
                        B[k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N],
                        B_shared[k % num_stages, :, :],
                        barrier=loaded[k % num_stages],
                    )
                # Load one packed uint32 SF word every sf_load_period iterations.
                if k % sf_load_period == 0:
                    sf_group_idx = k // sf_load_period
                    T.tma_copy(
                        SFA[sf_group_idx * M + bx * block_M : sf_group_idx * M + (bx + 1) * block_M],
                        SFA_shared[k % num_stages, :],
                        barrier=loaded[k % num_stages],
                    )
                    T.tma_copy(
                        SFB[sf_group_idx * N + by * block_N : sf_group_idx * N + (by + 1) * block_N],
                        SFB_shared[k % num_stages, :],
                        barrier=loaded[k % num_stages],
                    )
                T.mbarrier_arrive(loaded[k % num_stages])

        elif tx < 64:
            # Warp 1: MMA issue + UTCCP
            for k in T.serial(k_iters):
                stage = k % num_stages
                phase = (k // num_stages) & 1
                T.mbarrier_wait_parity(loaded[stage], phase)
                T.mbarrier_wait_parity(with_sf_full[stage], phase)

                if k % sf_load_period == 0:
                    T.tcgen05_cp_warpx4(SFA_shared[stage, :], SFA_tmem)
                    T.tcgen05_cp_warpx4(SFB_shared[stage, :], SFB_tmem)

                # sf_id selects which of the 4 packed E8M0 values to use
                T.tcgen05_gemm_blockscaled(
                    A_shared[stage, :, :],
                    B_shared[stage, :, :],
                    C_tmem,
                    SFA_tmem,
                    SFB_tmem,
                    transpose_B=transpose_B,
                    mbar=consumed[stage],
                    clear_accum=k == 0,
                    sf_a_id=k % sf_load_period,
                    sf_b_id=k % sf_load_period,
                )

            T.tcgen05_mma_arrive(tmem_full)

        elif tx < 96:
            # Warp 2: scale-factor transpose
            for k in T.serial(k_iters):
                stage = k % num_stages
                phase = (k // num_stages) & 1
                T.mbarrier_wait_parity(loaded[stage], phase)

                if k % sf_load_period == 0:
                    T.tcgen05_sf_warp_transpose(SFA_shared[stage, :])
                    T.tcgen05_sf_warp_transpose(SFB_shared[stage, :])
                    T.fence_proxy_async()
                T.mbarrier_arrive(with_sf_full[stage])

        # Epilogue: all warps
        T.mbarrier_wait_parity(tmem_full, 0)
        T.sync_threads()

        T.copy(C_tmem, C_local)
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[bx * block_M, by * block_N])

    return C


@tilelang.jit
def mxfp8_blockscaled_gemm_2cta(
    A,
    B,
    SFA,
    SFB,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    sf_granularity_k=128,
    transpose_B=False,
):
    M, N, K = T.const("M, N, K")

    assert block_M == 128
    assert block_N == 256
    assert block_K == 128
    assert sf_granularity_k == 128

    half_N = block_N // 2
    k_iters = T.ceildiv(K, block_K)
    sf_load_period = sf_granularity_k * 4 // block_K
    sf_k_groups = T.ceildiv(T.ceildiv(K, sf_granularity_k), 4)
    assert sf_load_period == 4

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[N, K] if transpose_B else [K, N], in_dtype]
    SFA: T.Tensor[[sf_k_groups * M], T.uint32]
    SFB: T.Tensor[[sf_k_groups * N], T.uint32]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128, cluster_dims=2) as (bx, by):
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)

        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared(
            (num_stages, half_N, block_K) if transpose_B else (num_stages, block_K, half_N),
            in_dtype,
        )
        SFA_shared = T.alloc_shared((num_stages, block_M), "uint32")
        SFB_shared = T.alloc_shared((num_stages, block_N), "uint32")

        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
        SFA_tmem = T.alloc_tmem([block_M, 4], "uint32")
        SFB_tmem = T.alloc_tmem([block_M, 8], "uint32")

        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        loaded = T.alloc_barrier([32] * num_stages)
        with_sf_full = T.alloc_cluster_barrier([32 * 2] * num_stages)
        consumed = T.alloc_cluster_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1])

        tx = T.get_thread_binding()
        warp_idx = tx // 32
        T.use_swizzle(16)

        if warp_idx == 0:
            for k in T.serial(k_iters):
                stage = k % num_stages
                phase = (k // num_stages) & 1
                T.mbarrier_wait_parity(consumed[stage], phase ^ 1)
                T.tma_copy(
                    A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                    A_shared[stage, :, :],
                    barrier=loaded[stage],
                )
                if transpose_B:
                    T.tma_copy(
                        B[
                            (by * block_N + cta_id * half_N) : (by * block_N + (cta_id + 1) * half_N),
                            k * block_K : (k + 1) * block_K,
                        ],
                        B_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                else:
                    T.tma_copy(
                        B[
                            k * block_K : (k + 1) * block_K,
                            (by * block_N + cta_id * half_N) : (by * block_N + (cta_id + 1) * half_N),
                        ],
                        B_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                if k % sf_load_period == 0:
                    sf_group_idx = k // sf_load_period
                    T.tma_copy(
                        SFA[sf_group_idx * M + bx * block_M : sf_group_idx * M + (bx + 1) * block_M],
                        SFA_shared[stage, :],
                        barrier=loaded[stage],
                    )
                    T.tma_copy(
                        SFB[sf_group_idx * N + by * block_N : sf_group_idx * N + (by + 1) * block_N],
                        SFB_shared[stage, :],
                        barrier=loaded[stage],
                    )
                T.mbarrier_arrive(loaded[stage])

        elif warp_idx == 1 and cta_id == 0:
            for k in T.serial(k_iters):
                stage = k % num_stages
                phase = (k // num_stages) & 1
                T.mbarrier_wait_parity(with_sf_full[stage], phase)
                if k % sf_load_period == 0:
                    T.tcgen05_cp_warpx4(SFA_shared[stage, :], SFA_tmem, use_2cta=True)
                    T.tcgen05_cp_warpx4(SFB_shared[stage, :], SFB_tmem, use_2cta=True)

                T.tcgen05_gemm_blockscaled(
                    A_shared[stage, :, :],
                    B_shared[stage, :, :],
                    C_tmem,
                    SFA_tmem,
                    SFB_tmem,
                    transpose_B=transpose_B,
                    mbar=consumed[stage],
                    clear_accum=k == 0,
                    sf_a_id=k % sf_load_period,
                    sf_b_id=k % sf_load_period,
                    use_2cta=True,
                )
            T.tcgen05_mma_arrive(tmem_full, arrive_2cta=True)

        elif warp_idx == 2:
            for k in T.serial(k_iters):
                stage = k % num_stages
                phase = (k // num_stages) & 1
                T.mbarrier_wait_parity(loaded[stage], phase)
                if k % sf_load_period == 0:
                    T.tcgen05_sf_warp_transpose(SFA_shared[stage, :])
                    T.tcgen05_sf_warp_transpose(SFB_shared[stage, :])
                    T.fence_proxy_async()
                T.mbarrier_arrive(with_sf_full[stage], 0)

        T.mbarrier_wait_parity(tmem_full, 0)
        T.copy(C_tmem, C_local)
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[bx * block_M, by * block_N])

    return C


@tilelang.jit
def mxfp8_blockscaled_gemm_2cta_persistent(
    A,
    B,
    SFA,
    SFB,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    sf_granularity_k=128,
    transpose_B=False,
    use_tma_store=True,
    store_block_N=64,
):
    M, N, K = T.const("M, N, K")

    half_N = block_N // 2
    k_iters = T.ceildiv(K, block_K)
    sf_load_period = sf_granularity_k * 4 // block_K
    sf_k_groups = T.ceildiv(T.ceildiv(K, sf_granularity_k), 4)

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[N, K] if transpose_B else [K, N], in_dtype]
    SFA: T.Tensor[[sf_k_groups * M], T.uint32]
    SFB: T.Tensor[[sf_k_groups * N], T.uint32]
    C = T.empty((M, N), out_dtype)

    sm_num = driver.get_num_sms()
    num_clusters = sm_num // 2
    m_blocks = T.ceildiv(M, block_M)
    m_clusters = m_blocks // 2
    n_blocks = T.ceildiv(N, block_N)
    assert K % (2 * block_K) == 0  # for simplicity
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 16  # in cluster
    assert n_blocks % (2 * group_size) == 0  # Please adjust group_size if not satisfied

    with T.Kernel(sm_num, threads=256, cluster_dims=2) as (block_id):
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)

        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared(
            (num_stages, half_N, block_K) if transpose_B else (num_stages, block_K, half_N),
            in_dtype,
        )
        SFA_shared = T.alloc_shared((num_stages, block_M), "uint32")
        SFB_shared = T.alloc_shared((num_stages, block_N), "uint32")

        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
        SFA_tmem = T.alloc_tmem([block_M, block_M // 128 * 4], "uint32")
        SFB_tmem = T.alloc_tmem([block_M, block_N // 128 * 4], "uint32")

        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)

        loaded = T.alloc_barrier([32] * num_stages)
        with_sf_full = T.alloc_cluster_barrier([32 * 2] * num_stages)
        consumed = T.alloc_cluster_barrier([1] * num_stages)
        tmem_full = T.alloc_cluster_barrier([1])
        tmem_empty = T.alloc_cluster_barrier([128 * 2])

        tx = T.get_thread_binding()
        warp_idx = tx // 32

        if warp_idx == 0:
            for w in T.unroll(waves):
                cluster_id = block_id // 2
                tile_id = num_clusters * w + cluster_id
                bx_cluster = (tile_id // group_size) % m_clusters
                bx = bx_cluster * 2 + cta_id
                by = (tile_id % group_size) + (tile_id // group_size) // m_clusters * group_size

                if bx * block_M < M and by * block_N < N:
                    for k in T.serial(k_iters):
                        phase = w * k_iters + k
                        stage = phase % num_stages
                        parity = (phase // num_stages) & 1
                        T.mbarrier_wait_parity(consumed[stage], parity ^ 1)
                        T.tma_copy(
                            A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                            A_shared[stage, :, :],
                            barrier=loaded[stage],
                        )
                        if transpose_B:
                            T.tma_copy(
                                B[
                                    by * block_N + cta_id * half_N : by * block_N + (cta_id + 1) * half_N,
                                    k * block_K : (k + 1) * block_K,
                                ],
                                B_shared[stage, :, :],
                                barrier=loaded[stage],
                            )
                        else:
                            T.tma_copy(
                                B[
                                    k * block_K : (k + 1) * block_K,
                                    by * block_N + cta_id * half_N : by * block_N + (cta_id + 1) * half_N,
                                ],
                                B_shared[stage, :, :],
                                barrier=loaded[stage],
                            )
                        if k % sf_load_period == 0:
                            sf_group_idx = k // sf_load_period
                            T.tma_copy(
                                SFA[sf_group_idx * M + bx * block_M : sf_group_idx * M + (bx + 1) * block_M],
                                SFA_shared[stage, :],
                                barrier=loaded[stage],
                            )
                            T.tma_copy(
                                SFB[sf_group_idx * N + by * block_N : sf_group_idx * N + (by + 1) * block_N],
                                SFB_shared[stage, :],
                                barrier=loaded[stage],
                            )
                        T.mbarrier_arrive(loaded[stage])

        elif warp_idx == 1 and cta_id == 0:
            for w in T.unroll(waves):
                cluster_id = block_id // 2
                tile_id = num_clusters * w + cluster_id
                bx_cluster = (tile_id // group_size) % m_clusters
                bx = bx_cluster * 2 + cta_id
                by = (tile_id % group_size) + (tile_id // group_size) // m_clusters * group_size

                if bx * block_M < M and by * block_N < N:
                    T.mbarrier_wait_parity(tmem_empty, (w & 1) ^ 1)
                    for k in T.serial(k_iters):
                        phase = w * k_iters + k
                        stage = phase % num_stages
                        parity = (phase // num_stages) & 1
                        T.mbarrier_wait_parity(with_sf_full[stage], parity)
                        if k % sf_load_period == 0:
                            T.tcgen05_cp_warpx4(SFA_shared[stage, :], SFA_tmem, use_2cta=True)
                            T.tcgen05_cp_warpx4(SFB_shared[stage, :], SFB_tmem, use_2cta=True)
                        T.tcgen05_gemm_blockscaled(
                            A_shared[stage, :, :],
                            B_shared[stage, :, :],
                            C_tmem,
                            SFA_tmem,
                            SFB_tmem,
                            transpose_B=transpose_B,
                            mbar=consumed[stage],
                            clear_accum=k == 0,
                            sf_a_id=k % sf_load_period,
                            sf_b_id=k % sf_load_period,
                            use_2cta=True,
                        )
                    T.tcgen05_mma_arrive(tmem_full, arrive_2cta=True)

        elif warp_idx == 2:
            for w in T.unroll(waves):
                cluster_id = block_id // 2
                tile_id = num_clusters * w + cluster_id
                bx_cluster = (tile_id // group_size) % m_clusters
                bx = bx_cluster * 2 + cta_id
                by = (tile_id % group_size) + (tile_id // group_size) // m_clusters * group_size

                if bx * block_M < M and by * block_N < N:
                    for k in T.serial(k_iters):
                        phase = w * k_iters + k
                        stage = phase % num_stages
                        parity = (phase // num_stages) & 1
                        T.mbarrier_wait_parity(loaded[stage], parity)
                        if k % sf_load_period == 0:
                            T.tcgen05_sf_warp_transpose(SFA_shared[stage, :])
                            T.tcgen05_sf_warp_transpose(SFB_shared[stage, :])
                            T.fence_proxy_async()
                        T.mbarrier_arrive(with_sf_full[stage], 0)

        elif 128 <= tx < 256:
            for w in T.unroll(waves):
                cluster_id = block_id // 2
                tile_id = num_clusters * w + cluster_id
                bx_cluster = (tile_id // group_size) % m_clusters
                bx = bx_cluster * 2 + cta_id
                by = (tile_id % group_size) + (tile_id // group_size) // m_clusters * group_size

                if bx * block_M < M and by * block_N < N:
                    T.mbarrier_wait_parity(tmem_full, w & 1)
                    T.copy(C_tmem, C_local)
                    T.mbarrier_arrive(tmem_empty, 0)

                    if use_tma_store:
                        for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                            T.copy(C_local[:, i * store_block_N : (i + 1) * store_block_N], C_shared)
                            T.copy(C_shared, C[bx * block_M, by * block_N + i * store_block_N])
                    else:
                        T.copy(C_local, C_local_cast)
                        T.copy(C_local_cast, C[bx * block_M, by * block_N])
    return C


def unpack_sf_u32_1d(packed_sf, mn, sf_k_blocks):
    sf_k_groups = (sf_k_blocks + 3) // 4
    packed_2d = packed_sf.view(sf_k_groups, mn).T.contiguous().to(torch.int64)
    unpacked = torch.empty((mn, sf_k_groups * 4), device=packed_sf.device, dtype=torch.uint8)
    for i in range(4):
        unpacked[:, i::4] = ((packed_2d >> (8 * i)) & 0xFF).to(torch.uint8)
    return unpacked[:, :sf_k_blocks].contiguous()


def pack_sf_u8_to_u32_1d(sf_u8):
    assert sf_u8.dtype == torch.uint8
    assert sf_u8.dim() == 2
    mn, sf_k_padded = sf_u8.shape
    assert sf_k_padded % 4 == 0
    words = sf_u8.to(torch.int64)
    packed = (words[:, 0::4] | (words[:, 1::4] << 8) | (words[:, 2::4] << 16) | (words[:, 3::4] << 24)).to(torch.uint32)
    return packed.T.contiguous().reshape(-1)


def quantize_fp8_with_packed_ue8m0(x, gran_k=128):
    """DeepGEMM-style per-token FP8 quantization with UE8M0 scale factors.

    Returns:
        x_fp8: [MN, K] in float8_e4m3fn
        sf_packed_u32: flattened group-major packed uint32 scale factors
        sf_u8: [MN, ceil(K / gran_k)] unpacked E8M0 exponents
    """

    def ceil_div_int(x, y):
        return (x + y - 1) // y

    def align_up(x, y):
        return ceil_div_int(x, y) * y

    def ceil_to_ue8m0(x):
        bits = x.abs().float().view(torch.int32)
        exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
        return (exp.clamp(1, 254) << 23).view(torch.float32)

    assert x.dim() == 2
    mn, k = x.shape
    padded_k = align_up(k, gran_k)

    x_padded = torch.zeros((mn, padded_k), device=x.device, dtype=x.dtype)
    x_padded[:, :k] = x
    x_view = x_padded.view(mn, padded_k // gran_k, gran_k)

    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)

    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.view(mn, padded_k)[:, :k].contiguous()

    sf_u8 = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    sf_k_blocks = sf_u8.shape[1]
    sf_k_padded = align_up(sf_k_blocks, 4)
    if sf_k_padded != sf_k_blocks:
        sf_u8_padded = torch.full((mn, sf_k_padded), 127, device=x.device, dtype=torch.uint8)
        sf_u8_padded[:, :sf_k_blocks] = sf_u8
    else:
        sf_u8_padded = sf_u8

    sf_packed_u32 = pack_sf_u8_to_u32_1d(sf_u8_padded)
    return x_fp8, sf_packed_u32, sf_u8


def blockscaled_gemm_ref(a, b, sfa_packed, sfb_packed, sf_granularity_k=128, transpose_B=False):
    """Torch reference for block-scaled MXFP8 GEMM.

    Args:
        a: [M, K] FP8 tensor
        b: [K, N] FP8 tensor, or [N, K] when transpose_B=True
        sfa_packed: [(sf_k_blocks / 4) * M] uint32 packed E8M0 scale factors for A
        sfb_packed: [(sf_k_blocks / 4) * N] uint32 packed E8M0 scale factors for B
        sf_granularity_k: number of K elements per scale factor block (default 128)

    Returns:
        [M, N] float32 result
    """
    M, K = a.shape
    if transpose_B:
        N, K2 = b.shape
    else:
        K2, N = b.shape
    assert K == K2
    sf_k_blocks = (K + sf_granularity_k - 1) // sf_granularity_k
    sfa_unpacked = unpack_sf_u32_1d(sfa_packed, M, sf_k_blocks)
    sfb_unpacked = unpack_sf_u32_1d(sfb_packed, N, sf_k_blocks)

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)

    # E8M0 exponent to float scale: 2^(exp - 127)
    sfa_scales = torch.pow(2.0, sfa_unpacked.to(torch.float32) - 127.0)  # [M, sf_k_blocks]
    sfb_scales = torch.pow(2.0, sfb_unpacked.to(torch.float32) - 127.0)  # [N, sf_k_blocks]

    c = torch.zeros(M, N, device=a.device, dtype=torch.float32)
    for bi in range(sf_k_blocks):
        k_start = bi * sf_granularity_k
        k_end = min(k_start + sf_granularity_k, K)
        # Scale A block: [M, block_k] * [M, 1]
        a_block = a_f32[:, k_start:k_end] * sfa_scales[:, bi : bi + 1]
        if transpose_B:
            # Scale B block: [N, block_k] * [N, 1]
            b_block = b_f32[:, k_start:k_end] * sfb_scales[:, bi : bi + 1]
            c += a_block @ b_block.T
        else:
            # Scale B block: [block_k, N] * [1, N]
            b_block = b_f32[k_start:k_end, :] * sfb_scales[:, bi : bi + 1].T
            c += a_block @ b_block
    return c


def cosine_similarity(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat) / (a_flat.norm() * b_flat.norm())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-e2e-quant-path", action="store_true", default=True)
    parser.add_argument("--persistent", action="store_true", default=True)
    parser.add_argument("--enable-2cta", action="store_true", default=True)
    parser.add_argument("--transpose-b", action="store_true", help="Use B as [N, K] and compute A @ B.T.")
    return parser.parse_args()


def main():
    args = parse_args()

    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K = 128, 256, 128
    in_dtype, out_dtype, accum_dtype = T.float8_e4m3fn, T.bfloat16, T.float
    use_e2e_quant_path = args.use_e2e_quant_path
    persistent = args.persistent
    enable_2cta = args.enable_2cta
    transpose_B = args.transpose_b
    num_stages = 6 if enable_2cta else 4
    if persistent:
        assert enable_2cta
        kernel = mxfp8_blockscaled_gemm_2cta_persistent
    else:
        kernel = mxfp8_blockscaled_gemm_2cta if enable_2cta else mxfp8_blockscaled_gemm
    sf_granularity_k = 128
    assert sf_granularity_k == 128

    if use_e2e_quant_path:
        # End-to-end path:
        #   fp16/bf16 source tensors -> per-token FP8 quantization with UE8M0 SF
        #   -> pack 4 SF entries into one uint32 -> blockscaled GEMM
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w_nt = torch.randn(N, K, device="cuda", dtype=torch.float16)

        a, sfa, _ = quantize_fp8_with_packed_ue8m0(x, gran_k=sf_granularity_k)
        b_nt, sfb, _ = quantize_fp8_with_packed_ue8m0(w_nt, gran_k=sf_granularity_k)
        b = b_nt if transpose_B else b_nt.T.contiguous()
    else:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
        if transpose_B:
            b = torch.randn(N, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
        else:
            b = torch.randn(K, N, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)

        # E8M0 scale factors: one uint32 per row per 4 K-blocks.
        sf_k_blocks = (K + sf_granularity_k - 1) // sf_granularity_k

        # Pad to multiple of 4 (UTCCP loads 4 K-blocks at a time)
        sf_k_padded = ((sf_k_blocks + 3) // 4) * 4
        sfa_u8 = torch.randint(127 - 5, 127 + 5, (M, sf_k_padded), device="cuda", dtype=torch.uint8)
        sfb_u8 = torch.randint(127 - 5, 127 + 5, (N, sf_k_padded), device="cuda", dtype=torch.uint8)
        sfa = pack_sf_u8_to_u32_1d(sfa_u8)
        sfb = pack_sf_u8_to_u32_1d(sfb_u8)

    c = kernel(
        a,
        b,
        sfa,
        sfb,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        sf_granularity_k,
        transpose_B,
    )
    print(
        kernel.get_kernel_source(
            a,
            b,
            sfa,
            sfb,
            block_M,
            block_N,
            block_K,
            in_dtype,
            out_dtype,
            accum_dtype,
            num_stages,
            sf_granularity_k,
            transpose_B,
        )
    )

    if use_e2e_quant_path:
        # For the end-to-end quantization path, compare against the reference with bf16 gemm
        ref_c = (x.float() @ w_nt.float().T).to(torch.bfloat16)
    else:
        ref_c = blockscaled_gemm_ref(a, b, sfa, sfb, sf_granularity_k, transpose_B=transpose_B).to(torch.bfloat16)
    sim = cosine_similarity(c, ref_c)

    print(f"Output shape: {c.shape}, dtype: {c.dtype}")
    print(f"E2E quant path: {use_e2e_quant_path}")
    print(f"transpose_B: {transpose_B}")
    print(f"{c=}, {ref_c=}")
    # print(f"Max abs error: {(c.float() - ref_c.float()).abs().max().item():.6f}")
    print(f"Cosine similarity: {sim.item():.6f}")
    if use_e2e_quant_path:
        assert 1 - sim < 1e-3  # err tolerance from DeepGEMM
        print("e2e check passed ✅")

    tl_latency = do_bench(
        lambda: kernel(
            a,
            b,
            sfa,
            sfb,
            block_M,
            block_N,
            block_K,
            in_dtype,
            out_dtype,
            accum_dtype,
            num_stages,
            sf_granularity_k,
            transpose_B,
        ),
        backend="cupti",
    )
    print(f"Tilelang MXFP8 latency: {tl_latency} ms")
    print(f"TFLOPs: {2 * M * N * K / (tl_latency / 1e3) / 1e12:.2f}")


if __name__ == "__main__":
    main()
