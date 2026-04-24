# MXFP8 Block-Scaled GEMM on SM100
# Blockscale size: (M, N, K) = (1, 1, 128)
# Explicit scale-factor path:
#   1. load packed uint32 scale factors to shared memory
#   2. transpose the 128-word tile in-place for UTCCP
#   3. issue tcgen05_cp to move the transposed tile into TMEM

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TILELANG_CACHE_DIR", "/tmp/tilelang-cache")
os.environ.setdefault("TILELANG_TMP_DIR", "/tmp/tilelang-cache/tmp")

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
):
    """1D-1D Block-scaled MXFP8 GEMM.

    A:   [M, K] in FP8 (E4M3 or E5M2)
    B:   [K, N] in FP8 (E4M3 or E5M2)
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
    B: T.Tensor[[K, N], in_dtype]
    SFA: T.Tensor[[sf_k_groups * M], T.uint32]
    SFB: T.Tensor[[sf_k_groups * N], T.uint32]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
        # Data shared memory (pipelined)
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N), in_dtype)

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
    B: T.Tensor[[K, N], in_dtype]
    SFA: T.Tensor[[sf_k_groups * M], T.uint32]
    SFB: T.Tensor[[sf_k_groups * N], T.uint32]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128, cluster_dims=2) as (bx, by):
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)

        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, half_N), in_dtype)
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
    use_tma_store=True,
    store_block_N=64,
):
    M, N, K = T.const("M, N, K")

    half_N = block_N // 2
    k_iters = T.ceildiv(K, block_K)
    sf_load_period = sf_granularity_k * 4 // block_K
    sf_k_groups = T.ceildiv(T.ceildiv(K, sf_granularity_k), 4)

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
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
        B_shared = T.alloc_shared((num_stages, block_K, half_N), in_dtype)
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
    packed = (
        words[:, 0::4]
        | (words[:, 1::4] << 8)
        | (words[:, 2::4] << 16)
        | (words[:, 3::4] << 24)
    ).to(torch.uint32)
    return packed.T.contiguous().reshape(-1)


def blockscaled_gemm_ref(a, b, sfa_packed, sfb_packed, sf_granularity_k=128):
    """Torch reference for block-scaled MXFP8 GEMM.

    Args:
        a: [M, K] FP8 tensor
        b: [K, N] FP8 tensor
        sfa_packed: [(sf_k_blocks / 4) * M] uint32 packed E8M0 scale factors for A
        sfb_packed: [(sf_k_blocks / 4) * N] uint32 packed E8M0 scale factors for B
        sf_granularity_k: number of K elements per scale factor block (default 128)

    Returns:
        [M, N] float32 result
    """
    M, K = a.shape
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
        # Scale B block: [block_k, N] * [1, N]  (sfb is [N, blocks], transpose for broadcast)
        b_block = b_f32[k_start:k_end, :] * sfb_scales[:, bi : bi + 1].T
        c += a_block @ b_block
    return c


def cosine_similarity(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat) / (a_flat.norm() * b_flat.norm())


def main():
    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K = 128, 256, 128
    in_dtype, out_dtype, accum_dtype = T.float8_e4m3fn, T.bfloat16, T.float
    persistent = True
    enable_2cta = True
    num_stages = 6 if enable_2cta else 4
    if persistent:
        assert enable_2cta, "2-CTA scheduling is required for the persistent version to achieve good performance"
        kernel = mxfp8_blockscaled_gemm_2cta_persistent
    else:
        kernel = mxfp8_blockscaled_gemm_2cta if enable_2cta else mxfp8_blockscaled_gemm
    sf_granularity_k = 128
    assert sf_granularity_k == 128

    a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
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
        )
    )

    ref_c = blockscaled_gemm_ref(a, b, sfa, sfb, sf_granularity_k).to(torch.bfloat16)
    sim = cosine_similarity(c, ref_c)
    print(f"Output shape: {c.shape}, dtype: {c.dtype}")
    print(f"{c=}, {ref_c=}")
    # print(f"Max abs error: {(c.float() - ref_c.float()).abs().max().item():.6f}")
    print(f"Cosine similarity: {sim.item():.6f}")

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
        ),
        backend="cupti",
    )
    print(f"Tilelang MXFP8 latency: {tl_latency} ms")
    print(f"TFLOPs: {2 * M * N * K / (tl_latency / 1e3) / 1e12:.2f}")


if __name__ == "__main__":
    main()
