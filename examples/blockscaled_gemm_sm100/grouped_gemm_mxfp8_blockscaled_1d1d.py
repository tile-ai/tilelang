# Grouped MXFP8 block-scaled GEMM on SM100.
# Blockscale size: (M, N, K) = (1, 1, 128)

import argparse

import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


@tilelang.jit
def grouped_mxfp8_blockscaled_gemm_2cta(
    A,
    B,
    SFA,
    SFB,
    offsets,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    max_M_per_E,
    transpose_B=True,
    sf_granularity_k=128,
):
    """Grouped 2CTA MXFP8 blockscaled GEMM.

    Logical scale shape follows tilelang_gemm.py:
      SFA [M_total, sf_k_packed], SFB [E, N, sf_k_packed]

    Kernel scale operands are group-major flat buffers so the SF loads can use
    the same contiguous TMA pattern as mxfp8_blockscaled_gemm_2cta.
    """
    M_total, N, K, E, E1 = T.const("M_total, N, K, E, E1")

    assert block_M == 128
    assert block_N == 256
    assert block_K == 128
    assert sf_granularity_k == 128

    half_N = block_N // 2
    k_iters = T.ceildiv(K, block_K)
    sf_load_period = sf_granularity_k * 4 // block_K
    sf_k_groups = T.ceildiv(T.ceildiv(K, sf_granularity_k), 4)
    assert sf_load_period == 4

    A: T.Tensor[[M_total, K], in_dtype]
    B: T.Tensor[[E, N, K] if transpose_B else [E, K, N], in_dtype]
    SFA: T.Tensor[[sf_k_groups * M_total], T.uint32]
    SFB: T.Tensor[[sf_k_groups * E * N], T.uint32]
    offsets: T.Tensor[[E1], T.int32]
    C = T.empty((M_total, N), out_dtype)

    n_blocks = T.ceildiv(N, block_N)
    max_M_blocks = T.ceildiv(max_M_per_E, block_M)
    max_M_blocks_padded = T.ceildiv(max_M_blocks, 2) * 2

    with T.Kernel(max_M_blocks_padded, n_blocks, E, threads=128, cluster_dims=2) as (pid_m, pid_n, eid):
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
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        loaded = T.alloc_barrier([32] * num_stages)
        with_sf_full = T.alloc_cluster_barrier([32 * 2] * num_stages)
        consumed = T.alloc_cluster_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1])

        tx = T.get_thread_binding()
        warp_idx = tx // 32
        T.use_swizzle(16)

        start_m = offsets[eid]
        end_m = offsets[eid + 1]
        m_size = end_m - start_m
        expert_m_blocks = T.ceildiv(m_size, block_M)
        clamped_pid_m = T.min(pid_m, T.max(expert_m_blocks, 1) - 1)
        tile_m = start_m + clamped_pid_m * block_M

        if warp_idx == 0:
            for k in T.serial(k_iters):
                stage = k % num_stages
                phase = (k // num_stages) & 1
                T.mbarrier_wait_parity(consumed[stage], phase ^ 1)
                T.tma_copy(
                    A[tile_m : tile_m + block_M, k * block_K : (k + 1) * block_K],
                    A_shared[stage, :, :],
                    barrier=loaded[stage],
                )
                if transpose_B:
                    T.tma_copy(
                        B[
                            eid,
                            pid_n * block_N + cta_id * half_N : pid_n * block_N + (cta_id + 1) * half_N,
                            k * block_K : (k + 1) * block_K,
                        ],
                        B_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                else:
                    T.tma_copy(
                        B[
                            eid,
                            k * block_K : (k + 1) * block_K,
                            pid_n * block_N + cta_id * half_N : pid_n * block_N + (cta_id + 1) * half_N,
                        ],
                        B_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                if k % sf_load_period == 0:
                    sf_group_idx = k // sf_load_period
                    T.tma_copy(
                        SFA[sf_group_idx * M_total + tile_m : sf_group_idx * M_total + tile_m + block_M],
                        SFA_shared[stage, :],
                        barrier=loaded[stage],
                    )
                    T.tma_copy(
                        SFB[sf_group_idx * E * N + eid * N + pid_n * block_N : sf_group_idx * E * N + eid * N + (pid_n + 1) * block_N],
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

        if pid_m * block_M < m_size and tile_m + block_M <= end_m:
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[tile_m, pid_n * block_N])
        elif pid_m * block_M < m_size:
            T.copy(C_local, C_local_cast)
            actual_rows = end_m - tile_m
            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows and pid_n * block_N + j < N:
                    C[tile_m + i, pid_n * block_N + j] = C_local_cast[i, j]

    return C


@tilelang.jit
def grouped_mxfp8_blockscaled_gemm_2cta_persistent(
    A,
    B,
    SFA,
    SFB,
    offsets,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    max_M_per_E,
    transpose_B=True,
    sf_granularity_k=128,
    store_block_N=64,
):
    """Persistent grouped 2CTA MXFP8 blockscaled GEMM with one accumulator TMEM."""
    M_total, N, K, E, E1 = T.const("M_total, N, K, E, E1")

    assert block_M == 128
    assert block_N == 256
    assert block_K == 128
    assert sf_granularity_k == 128

    half_N = block_N // 2
    k_iters = T.ceildiv(K, block_K)
    sf_load_period = sf_granularity_k * 4 // block_K
    sf_k_groups = T.ceildiv(T.ceildiv(K, sf_granularity_k), 4)
    assert sf_load_period == 4

    A: T.Tensor[[M_total, K], in_dtype]
    B: T.Tensor[[E, N, K] if transpose_B else [E, K, N], in_dtype]
    SFA: T.Tensor[[sf_k_groups * M_total], T.uint32]
    SFB: T.Tensor[[sf_k_groups * E * N], T.uint32]
    offsets: T.Tensor[[E1], T.int32]
    C = T.empty((M_total, N), out_dtype)

    sm_num = driver.get_num_sms()
    num_clusters = sm_num // 2
    n_blocks = T.ceildiv(N, block_N)
    max_M_blocks = T.ceildiv(max_M_per_E, block_M)
    max_M_blocks_padded = T.ceildiv(max_M_blocks, 2) * 2
    m_clusters = max_M_blocks_padded // 2
    total_cluster_tiles = E * n_blocks * m_clusters
    waves = T.ceildiv(total_cluster_tiles, num_clusters)
    group_size = 8

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
                eid = tile_id // (n_blocks * m_clusters)
                local_tile_id = tile_id - eid * n_blocks * m_clusters
                num_pid_in_group = group_size * n_blocks
                group_id = local_tile_id // num_pid_in_group
                first_pid_m_cluster = group_id * group_size
                group_m = T.min(m_clusters - first_pid_m_cluster, group_size)
                pid_m_cluster = first_pid_m_cluster + (local_tile_id % num_pid_in_group) % group_m
                pid_n = (local_tile_id % num_pid_in_group) // group_m

                if tile_id < total_cluster_tiles:
                    start_m = offsets[eid]
                    end_m = offsets[eid + 1]
                    m_size = end_m - start_m
                    expert_m_blocks = T.ceildiv(m_size, block_M)
                    pid_m = pid_m_cluster * 2 + cta_id
                    safe_pid_m = T.min(pid_m, T.max(expert_m_blocks, 1) - 1)
                    tile_m = start_m + safe_pid_m * block_M

                    for k in T.serial(k_iters):
                        phase = w * k_iters + k
                        stage = phase % num_stages
                        parity = (phase // num_stages) & 1
                        T.mbarrier_wait_parity(consumed[stage], parity ^ 1)
                        T.tma_copy(
                            A[tile_m : tile_m + block_M, k * block_K : (k + 1) * block_K],
                            A_shared[stage, :, :],
                            barrier=loaded[stage],
                        )
                        if transpose_B:
                            T.tma_copy(
                                B[
                                    eid,
                                    pid_n * block_N + cta_id * half_N : pid_n * block_N + (cta_id + 1) * half_N,
                                    k * block_K : (k + 1) * block_K,
                                ],
                                B_shared[stage, :, :],
                                barrier=loaded[stage],
                            )
                        else:
                            T.tma_copy(
                                B[
                                    eid,
                                    k * block_K : (k + 1) * block_K,
                                    pid_n * block_N + cta_id * half_N : pid_n * block_N + (cta_id + 1) * half_N,
                                ],
                                B_shared[stage, :, :],
                                barrier=loaded[stage],
                            )
                        if k % sf_load_period == 0:
                            sf_group_idx = k // sf_load_period
                            T.tma_copy(
                                SFA[sf_group_idx * M_total + tile_m : sf_group_idx * M_total + tile_m + block_M],
                                SFA_shared[stage, :],
                                barrier=loaded[stage],
                            )
                            T.tma_copy(
                                SFB[
                                    sf_group_idx * E * N + eid * N + pid_n * block_N : sf_group_idx * E * N
                                    + eid * N
                                    + (pid_n + 1) * block_N
                                ],
                                SFB_shared[stage, :],
                                barrier=loaded[stage],
                            )
                        T.mbarrier_arrive(loaded[stage])

        elif warp_idx == 1 and cta_id == 0:
            for w in T.unroll(waves):
                cluster_id = block_id // 2
                tile_id = num_clusters * w + cluster_id

                if tile_id < total_cluster_tiles:
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

                if tile_id < total_cluster_tiles:
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
                eid = tile_id // (n_blocks * m_clusters)
                local_tile_id = tile_id - eid * n_blocks * m_clusters
                num_pid_in_group = group_size * n_blocks
                group_id = local_tile_id // num_pid_in_group
                first_pid_m_cluster = group_id * group_size
                group_m = T.min(m_clusters - first_pid_m_cluster, group_size)
                pid_m_cluster = first_pid_m_cluster + (local_tile_id % num_pid_in_group) % group_m
                pid_n = (local_tile_id % num_pid_in_group) // group_m
                pid_m = pid_m_cluster * 2 + cta_id

                if tile_id < total_cluster_tiles:
                    start_m = offsets[eid]
                    end_m = offsets[eid + 1]
                    m_size = end_m - start_m
                    tile_m = start_m + pid_m * block_M
                    T.mbarrier_wait_parity(tmem_full, w & 1)
                    T.copy(C_tmem, C_local)
                    T.mbarrier_arrive(tmem_empty, 0)

                    if pid_m * block_M < m_size and tile_m + block_M <= end_m:
                        for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                            T.copy(C_local[:, i * store_block_N : (i + 1) * store_block_N], C_shared)
                            T.copy(C_shared, C[tile_m, pid_n * block_N + i * store_block_N])
                    elif pid_m * block_M < m_size:
                        T.copy(C_local, C_local_cast)
                        actual_rows = end_m - tile_m
                        for i, j in T.Parallel(block_M, block_N):
                            if i < actual_rows and pid_n * block_N + j < N:
                                C[tile_m + i, pid_n * block_N + j] = C_local_cast[i, j]

    return C


def pack_sf_u8_to_u32_rows(sf_u8):
    assert sf_u8.dtype == torch.uint8
    assert sf_u8.dim() == 2
    assert sf_u8.shape[1] % 4 == 0
    words = sf_u8.to(torch.int64)
    return (words[:, 0::4] | (words[:, 1::4] << 8) | (words[:, 2::4] << 16) | (words[:, 3::4] << 24)).to(torch.uint32).contiguous()


def pack_rows_to_group_major_flat(packed_rows):
    return packed_rows.contiguous().T.contiguous().reshape(-1)


def pack_sfb_to_group_major_flat(packed_sfb):
    return packed_sfb.contiguous().permute(2, 0, 1).contiguous().reshape(-1)


def unpack_sf_u32_rows(packed_sf, sf_k_blocks):
    words = packed_sf.contiguous().view(-1, packed_sf.shape[-1]).to(torch.int64)
    unpacked = torch.empty((words.shape[0], words.shape[1] * 4), device=packed_sf.device, dtype=torch.uint8)
    for i in range(4):
        unpacked[:, i::4] = ((words >> (8 * i)) & 0xFF).to(torch.uint8)
    return unpacked[:, :sf_k_blocks].view(*packed_sf.shape[:-1], sf_k_blocks).contiguous()


def quantize_fp8_with_packed_ue8m0_rows(x, gran_k=128):
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
    sf_k_padded = align_up(sf_u8.shape[1], 4)
    if sf_k_padded != sf_u8.shape[1]:
        sf_u8_padded = torch.full((mn, sf_k_padded), 127, device=x.device, dtype=torch.uint8)
        sf_u8_padded[:, : sf_u8.shape[1]] = sf_u8
    else:
        sf_u8_padded = sf_u8
    return x_fp8, pack_sf_u8_to_u32_rows(sf_u8_padded), sf_u8


def grouped_blockscaled_gemm_ref(a, b, sfa_packed, sfb_packed, offsets, sf_granularity_k=128, transpose_B=True):
    m_total, k = a.shape
    if transpose_B:
        e, n, k2 = b.shape
    else:
        e, k2, n = b.shape
    assert k == k2
    sf_k_blocks = (k + sf_granularity_k - 1) // sf_granularity_k
    sfa_unpacked = unpack_sf_u32_rows(sfa_packed, sf_k_blocks)
    sfb_unpacked = unpack_sf_u32_rows(sfb_packed, sf_k_blocks)

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    sfa_scales = torch.pow(2.0, sfa_unpacked.to(torch.float32) - 127.0)
    sfb_scales = torch.pow(2.0, sfb_unpacked.to(torch.float32) - 127.0)

    c = torch.empty((m_total, n), device=a.device, dtype=torch.float32)
    for eid in range(e):
        start = int(offsets[eid].item())
        end = int(offsets[eid + 1].item())
        if start == end:
            continue
        out = torch.zeros((end - start, n), device=a.device, dtype=torch.float32)
        for bi in range(sf_k_blocks):
            k_start = bi * sf_granularity_k
            k_end = min(k_start + sf_granularity_k, k)
            a_block = a_f32[start:end, k_start:k_end] * sfa_scales[start:end, bi : bi + 1]
            if transpose_B:
                b_block = b_f32[eid, :, k_start:k_end] * sfb_scales[eid, :, bi : bi + 1]
                out += a_block @ b_block.T
            else:
                b_block = b_f32[eid, k_start:k_end, :] * sfb_scales[eid, :, bi : bi + 1].T
                out += a_block @ b_block
        c[start:end] = out
    return c


def cosine_similarity(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat) / (a_flat.norm() * b_flat.norm())


def make_offsets(batch_sizes, device):
    offsets = torch.zeros(len(batch_sizes) + 1, device=device, dtype=torch.int32)
    offsets[1:] = torch.tensor(batch_sizes, device=device, dtype=torch.int32).cumsum(0)
    return offsets


def run_grouped_mxfp8_blockscaled_gemm(
    a,
    b,
    sfa_flat,
    sfb_flat,
    offsets,
    max_M_per_E,
    transpose_B=True,
    persistent=True,
):
    block_M, block_N, block_K = 128, 256, 128
    in_dtype, out_dtype, accum_dtype = T.float8_e4m3fn, T.bfloat16, T.float
    num_stages = 6
    sf_granularity_k = 128

    m_total, k = a.shape
    if transpose_B:
        _, n, k2 = b.shape
    else:
        _, k2, n = b.shape
    assert k == k2
    assert n % block_N == 0, f"N={n} not divisible by {block_N}"
    assert k % block_K == 0, f"K={k} not divisible by {block_K}"

    kernel = grouped_mxfp8_blockscaled_gemm_2cta_persistent if persistent else grouped_mxfp8_blockscaled_gemm_2cta
    return kernel(
        a,
        b,
        sfa_flat,
        sfb_flat,
        offsets,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        max_M_per_E,
        transpose_B,
        sf_granularity_k,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=str, default="512,1024,1536,2048")
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--transpose-b", action="store_true", help="Use B as [E, N, K] and compute grouped A @ B.T.")
    parser.add_argument("--no-persistent", action="store_true", help="Run the non-persistent 2CTA kernel.")
    parser.add_argument("--no-bench", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    transpose_B = args.transpose_b
    persistent = not args.no_persistent
    device = "cuda"
    m_total = sum(batch_sizes)
    e = len(batch_sizes)
    n = args.N
    k = args.K

    offsets = make_offsets(batch_sizes, device)
    max_M_per_E = max(batch_sizes)

    x = torch.randn(m_total, k, device=device, dtype=torch.float16)
    w_nt = torch.randn(e, n, k, device=device, dtype=torch.float16)

    a, sfa, _ = quantize_fp8_with_packed_ue8m0_rows(x)
    b_nt, sfb_2d, _ = quantize_fp8_with_packed_ue8m0_rows(w_nt.view(e * n, k))
    b_nt = b_nt.view(e, n, k).contiguous()
    sfb = sfb_2d.view(e, n, -1).contiguous()

    sfa_flat = pack_rows_to_group_major_flat(sfa)
    sfb_flat = pack_sfb_to_group_major_flat(sfb)
    b = b_nt if transpose_B else b_nt.transpose(1, 2).contiguous()

    c = run_grouped_mxfp8_blockscaled_gemm(
        a,
        b,
        sfa_flat,
        sfb_flat,
        offsets,
        max_M_per_E,
        transpose_B,
        persistent,
    )
    ref_c = grouped_blockscaled_gemm_ref(a, b, sfa, sfb, offsets, transpose_B=transpose_B).to(torch.bfloat16)
    sim = cosine_similarity(c, ref_c)
    max_abs = (c.float() - ref_c.float()).abs().max().item()

    print(f"Output shape: {c.shape}, dtype: {c.dtype}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"transpose_B: {transpose_B}")
    print(f"persistent: {persistent}")
    print(f"Cosine similarity: {sim.item():.6f}")
    print(f"Max abs error: {max_abs:.6f}")
    assert 1 - sim < 1e-5
    print("grouped blockscaled check passed")

    if not args.no_bench:
        latency = do_bench(
            lambda: run_grouped_mxfp8_blockscaled_gemm(
                a,
                b,
                sfa_flat,
                sfb_flat,
                offsets,
                max_M_per_E,
                transpose_B,
                persistent,
            ),
            backend="cupti",
        )
        print(f"Tilelang grouped MXFP8 latency: {latency} ms")
        print(f"TFLOPs: {2 * m_total * n * k / (latency / 1e3) / 1e12:.2f}")


if __name__ == "__main__":
    main()
