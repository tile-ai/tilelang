import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver


def _torch_logits_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported logits dtype: {dtype}")


def _tilelang_logits_dtype(dtype: str):
    if dtype == "float32":
        return T.float32
    if dtype == "bfloat16":
        return T.bfloat16
    raise ValueError(f"unsupported logits dtype: {dtype}")


@dataclass(frozen=True)
class MQALogitsConfig:
    seq_len: int = 2048
    seq_len_kv: int = 4096
    num_heads: int = 64
    head_dim: int = 128
    logits_dtype: str = "float32"
    seed: int = 0

    @property
    def block_q(self) -> int:
        return 128 // self.num_heads

    def validate(self) -> None:
        assert self.num_heads == 64, "SM100 MQA SOTA kernels currently require num_heads=64"
        assert self.head_dim == 128, "SM100 MQA SOTA kernels currently require head_dim=128"
        assert self.seq_len > 0 and self.seq_len_kv > 0, "sequence lengths must be positive"
        assert self.seq_len <= self.seq_len_kv, (
            "seq_len must be <= seq_len_kv for the demo causal ranges"
        )
        assert self.seq_len_kv - self.seq_len >= 128, (
            "seq_len_kv must exceed seq_len by at least one 128-wide tile"
        )
        assert self.seq_len % self.block_q == 0, "seq_len must be divisible by block_q"
        assert self.seq_len_kv % 128 == 0, "seq_len_kv must be divisible by 128"
        assert self.logits_dtype in (
            "float32",
            "bfloat16",
        ), f"unsupported logits dtype: {self.logits_dtype}"


def generate_ks_ke(config: MQALogitsConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    ks = torch.zeros(config.seq_len, dtype=torch.int32, device="cuda")
    ke = torch.arange(config.seq_len, dtype=torch.int32, device="cuda")
    ke = ke + (config.seq_len_kv - config.seq_len)
    return ks, ke


def ref_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    seq_len_kv = kv.shape[0]
    q_f32 = q.float()
    kv_f32 = kv.float()
    cols = torch.arange(seq_len_kv, device=q.device)
    logits = torch.empty((q.shape[0], seq_len_kv), device=q.device, dtype=torch.float32)
    chunk = 128
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        score = torch.einsum("mhd,nd->hmn", q_f32[start:end], kv_f32)
        part = (score.relu() * weights[start:end].unsqueeze(-1).transpose(0, 1)).sum(dim=0)
        mask = (cols[None, :] >= ks[start:end, None]) & (cols[None, :] < ke[start:end, None])
        logits[start:end] = part.masked_fill(~mask, float("-inf"))
    return logits


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
)
def mqa_logits_fp4_persistent_ws_kernel(
    Q,
    QScale,
    KV,
    KVScale,
    Weights,
    KS,
    KE,
    Logits,
    seq_len: int,
    seq_len_kv: int,
    heads: int = 64,
    head_dim: int = 128,
    logits_stride: int = 4096,
    compressed_logits: bool = False,
    logits_dtype=T.float32,
):
    block_q = 128 // heads
    block_kv = 256
    half_kv = 128
    # The two KV halves are consecutive issues in a 2-phase TMEM stage ring.
    # Flip phase before wrapping stage: eager lowering guards each update separately.
    tmem_stage_step = block_kv // half_kv
    num_q_stages = 3
    num_stages = 5
    num_tmem_stages = 3
    sf_granularity_k = 32
    sf_k_groups = T.ceildiv(T.ceildiv(head_dim, sf_granularity_k), 4)
    accum_dtype = T.float32
    num_q_blocks = T.ceildiv(seq_len, block_q)
    sm_num = driver.get_num_sms()

    Q: T.Tensor((seq_len * heads, head_dim), T.float4_e2m1fn)
    QScale: T.Tensor((sf_k_groups * seq_len * heads,), T.uint32)
    KV: T.Tensor((seq_len_kv, head_dim), T.float4_e2m1fn)
    KVScale: T.Tensor((sf_k_groups * seq_len_kv,), T.uint32)
    Weights: T.Tensor((seq_len, heads), accum_dtype)
    KS: T.Tensor((seq_len,), T.int32)
    KE: T.Tensor((seq_len,), T.int32)
    Logits: T.Tensor((seq_len, logits_stride), logits_dtype)

    with T.Kernel(sm_num, threads=384) as block_id:
        q_shared = T.alloc_shared((num_q_stages, block_q * heads, head_dim), T.float4_e2m1_unpacked)
        sf_q_shared = T.alloc_shared((num_q_stages, block_q * heads), T.uint32)
        weights_shared = T.alloc_shared((num_q_stages, block_q, heads), accum_dtype)
        kv_shared_0 = T.alloc_shared((num_stages, half_kv, head_dim), T.float4_e2m1_unpacked)
        kv_shared_1 = T.alloc_shared((num_stages, half_kv, head_dim), T.float4_e2m1_unpacked)
        sf_kv_shared_0 = T.alloc_shared((num_stages, half_kv), T.uint32)
        sf_kv_shared_1 = T.alloc_shared((num_stages, half_kv), T.uint32)
        c_tmem = T.alloc_tmem((half_kv, 512), accum_dtype)
        sf_q_col_0 = block_q * heads * num_tmem_stages
        sf_q_col_1 = sf_q_col_0 + 4
        sf_kv_col_0 = sf_q_col_1 + 4
        sf_kv_col_1 = sf_kv_col_0 + 4
        q_loaded = T.alloc_barrier([32] * num_q_stages)
        q_empty = T.alloc_barrier([352] * num_q_stages)
        q_sf_full = T.alloc_barrier([32] * num_q_stages)
        kv_loaded = T.alloc_barrier([32] * num_stages)
        kv_sf_full_0 = T.alloc_barrier([32] * num_stages)
        kv_sf_full_1 = T.alloc_barrier([32] * num_stages)
        kv_empty = T.alloc_barrier([64] * num_stages)
        tmem_full = T.alloc_barrier([1] * num_tmem_stages)
        tmem_empty = T.alloc_barrier([128] * num_tmem_stages)

        tx = T.get_thread_binding()

        if tx < 32:
            T.dec_max_nreg(56)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            tile_iter = T.alloc_var(T.int32, init=0)
            if q_block < num_q_blocks:
                first_q_row = q_block * block_q
                T.tma_copy(
                    Q[first_q_row * heads : first_q_row * heads + block_q * heads, :],
                    q_shared[0, :, :],
                    barrier=q_loaded[0],
                )
                T.tma_copy(
                    QScale[first_q_row * heads : first_q_row * heads + block_q * heads],
                    sf_q_shared[0, :],
                    barrier=q_loaded[0],
                )
                T.tma_copy(
                    Weights[first_q_row : first_q_row + block_q, :],
                    weights_shared[0, :, :],
                    barrier=q_loaded[0],
                )
                T.mbarrier_arrive(q_loaded[0])
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset in T.unroll(block_q - 1):
                    qi = qi_offset + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                next_q_block = q_block + sm_num
                next_q_iter = q_iter + 1
                next_q_stage = next_q_iter % num_q_stages
                next_q_phase = (next_q_iter // num_q_stages) & 1
                if next_q_block < num_q_blocks:
                    T.mbarrier_wait_parity(q_empty[next_q_stage], next_q_phase ^ 1)
                    next_q_row = next_q_block * block_q
                    T.tma_copy(
                        Q[next_q_row * heads : next_q_row * heads + block_q * heads, :],
                        q_shared[next_q_stage, :, :],
                        barrier=q_loaded[next_q_stage],
                    )
                    T.tma_copy(
                        QScale[next_q_row * heads : next_q_row * heads + block_q * heads],
                        sf_q_shared[next_q_stage, :],
                        barrier=q_loaded[next_q_stage],
                    )
                    T.tma_copy(
                        Weights[next_q_row : next_q_row + block_q, :],
                        weights_shared[next_q_stage, :, :],
                        barrier=q_loaded[next_q_stage],
                    )
                    T.mbarrier_arrive(q_loaded[next_q_stage])

                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    kv_row = (first_bkv + kv_iter) * block_kv
                    stage = tile_iter % num_stages
                    parity = (tile_iter // num_stages) & 1
                    T.mbarrier_wait_parity(kv_empty[stage], parity ^ 1)
                    T.tma_copy(KV[kv_row : kv_row + half_kv, :], kv_shared_0[stage, :, :], barrier=kv_loaded[stage])
                    T.tma_copy(
                        KV[kv_row + half_kv : kv_row + block_kv, :],
                        kv_shared_1[stage, :, :],
                        barrier=kv_loaded[stage],
                    )
                    T.tma_copy(KVScale[kv_row : kv_row + half_kv], sf_kv_shared_0[stage, :], barrier=kv_loaded[stage])
                    T.tma_copy(
                        KVScale[kv_row + half_kv : kv_row + block_kv],
                        sf_kv_shared_1[stage, :],
                        barrier=kv_loaded[stage],
                    )
                    T.mbarrier_arrive(kv_loaded[stage])
                    tile_iter = tile_iter + 1
                    kv_iter = kv_iter + 1

                q_block = next_q_block
                q_iter = next_q_iter

        elif 32 <= tx < 64:
            T.dec_max_nreg(56)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            tile_iter = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset in T.unroll(block_q - 1):
                    qi = qi_offset + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                T.tcgen05_sf_warp_transpose(sf_q_shared[q_stage, :])
                T.fence_proxy_async()
                T.mbarrier_arrive(q_sf_full[q_stage])
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    stage = tile_iter % num_stages
                    parity = (tile_iter // num_stages) & 1
                    T.mbarrier_wait_parity(kv_loaded[stage], parity)
                    T.tcgen05_sf_warp_transpose(sf_kv_shared_0[stage, :])
                    T.fence_proxy_async()
                    T.mbarrier_arrive(kv_sf_full_0[stage])
                    T.tcgen05_sf_warp_transpose(sf_kv_shared_1[stage, :])
                    T.fence_proxy_async()
                    T.mbarrier_arrive(kv_sf_full_1[stage])
                    tile_iter = tile_iter + 1
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 64 <= tx < 96:
            T.dec_max_nreg(56)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            tile_iter = T.alloc_var(T.int32, init=0)
            tmem_stage = T.alloc_var(T.int32, init=0)
            tmem_phase = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset in T.unroll(block_q - 1):
                    qi = qi_offset + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                T.mbarrier_wait_parity(q_sf_full[q_stage], q_phase)
                T.tcgen05_cp_warpx4(sf_q_shared[q_stage, :], c_tmem, tmem_col_offset=sf_q_col_0)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    stage = tile_iter % num_stages
                    parity = (tile_iter // num_stages) & 1
                    tmem_col = tmem_stage * block_q * heads
                    T.mbarrier_wait_parity(tmem_empty[tmem_stage], tmem_phase ^ 1)
                    T.mbarrier_wait_parity(kv_loaded[stage], parity)
                    T.mbarrier_wait_parity(kv_sf_full_0[stage], parity)
                    T.tcgen05_cp_warpx4(sf_kv_shared_0[stage, :], c_tmem, tmem_col_offset=sf_kv_col_0)
                    T.tcgen05_gemm_blockscaled(
                        kv_shared_0[stage, :, :],
                        q_shared[q_stage, :, :],
                        c_tmem[:, tmem_col : tmem_col + block_q * heads],
                        c_tmem[:, sf_kv_col_0 : sf_kv_col_0 + 4],
                        c_tmem[:, sf_q_col_0 : sf_q_col_0 + 4],
                        transpose_B=True,
                        mbar=tmem_full[tmem_stage],
                        clear_accum=True,
                        k_start=0,
                        sf_a_granularity_k=sf_granularity_k,
                        sf_b_granularity_k=sf_granularity_k,
                    )
                    T.mbarrier_arrive(kv_empty[stage])
                    tile_iter = tile_iter + 1
                    tmem_stage = tmem_stage + tmem_stage_step
                    if tmem_stage >= num_tmem_stages:
                        tmem_phase = tmem_phase ^ 1
                        tmem_stage = tmem_stage - num_tmem_stages
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 96 <= tx < 128:
            T.dec_max_nreg(56)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            tile_iter = T.alloc_var(T.int32, init=0)
            tmem_stage = T.alloc_var(T.int32, init=1)
            tmem_phase = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset in T.unroll(block_q - 1):
                    qi = qi_offset + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                T.mbarrier_wait_parity(q_sf_full[q_stage], q_phase)
                T.tcgen05_cp_warpx4(sf_q_shared[q_stage, :], c_tmem, tmem_col_offset=sf_q_col_1)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    stage = tile_iter % num_stages
                    parity = (tile_iter // num_stages) & 1
                    tmem_col = tmem_stage * block_q * heads
                    T.mbarrier_wait_parity(tmem_empty[tmem_stage], tmem_phase ^ 1)
                    T.mbarrier_wait_parity(kv_loaded[stage], parity)
                    T.mbarrier_wait_parity(kv_sf_full_1[stage], parity)
                    T.tcgen05_cp_warpx4(sf_kv_shared_1[stage, :], c_tmem, tmem_col_offset=sf_kv_col_1)
                    T.tcgen05_gemm_blockscaled(
                        kv_shared_1[stage, :, :],
                        q_shared[q_stage, :, :],
                        c_tmem[:, tmem_col : tmem_col + block_q * heads],
                        c_tmem[:, sf_kv_col_1 : sf_kv_col_1 + 4],
                        c_tmem[:, sf_q_col_1 : sf_q_col_1 + 4],
                        transpose_B=True,
                        mbar=tmem_full[tmem_stage],
                        clear_accum=True,
                        k_start=0,
                        sf_a_granularity_k=sf_granularity_k,
                        sf_b_granularity_k=sf_granularity_k,
                    )
                    T.mbarrier_arrive(kv_empty[stage])
                    tile_iter = tile_iter + 1
                    tmem_stage = tmem_stage + tmem_stage_step
                    if tmem_stage >= num_tmem_stages:
                        tmem_phase = tmem_phase ^ 1
                        tmem_stage = tmem_stage - num_tmem_stages
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 128 <= tx < 256:
            T.inc_max_nreg(224)
            c_epi0 = T.alloc_fragment((half_kv, 16), accum_dtype)
            logits_epi0 = T.alloc_fragment((half_kv,), accum_dtype)
            weights_epi0 = T.alloc_fragment((block_q, heads), accum_dtype)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            tmem_stage = T.alloc_var(T.int32, init=0)
            tmem_phase = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_epi0 in T.unroll(block_q - 1):
                    qi_scan_epi0 = qi_offset_epi0 + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_epi0])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_epi0])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                if num_kv_blocks > 0:
                    T.copy(weights_shared[q_stage, :, :], weights_epi0)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    kv_row = (first_bkv + kv_iter) * block_kv
                    tmem_col = tmem_stage * block_q * heads
                    T.assume(tmem_col >= 0)
                    T.assume(tmem_col + block_q * heads <= block_q * heads * num_tmem_stages)
                    T.mbarrier_wait_parity(tmem_full[tmem_stage], tmem_phase)
                    for qi_epi0 in T.unroll(block_q):
                        for bn_init_epi0 in T.Parallel(half_kv):
                            logits_epi0[bn_init_epi0] = T.float32(0)
                        for h_base_epi0 in T.unroll(heads // 16):
                            T.copy(
                                c_tmem[
                                    :,
                                    tmem_col
                                    + qi_epi0 * heads
                                    + h_base_epi0 * 16 : tmem_col
                                    + qi_epi0 * heads
                                    + (h_base_epi0 + 1) * 16,
                                ],
                                c_epi0,
                            )
                            for bn_reduce_epi0 in T.Parallel(half_kv):
                                for h_inner_epi0 in T.vectorized(16):
                                    h_epi0 = h_base_epi0 * 16 + h_inner_epi0
                                    logits_epi0[bn_reduce_epi0] += (
                                        T.max(
                                            c_epi0[bn_reduce_epi0, h_inner_epi0],
                                            T.float32(0),
                                        )
                                        * weights_epi0[qi_epi0, h_epi0]
                                    )
                        for bn_store_epi0 in T.Parallel(half_kv):
                            Logits[q_row + qi_epi0, kv_row + bn_store_epi0] = T.cast(
                                logits_epi0[bn_store_epi0], logits_dtype
                            )
                        T.sync_warp()
                    T.mbarrier_arrive(tmem_empty[tmem_stage])
                    tmem_stage = tmem_stage + tmem_stage_step
                    if tmem_stage >= num_tmem_stages:
                        tmem_phase = tmem_phase ^ 1
                        tmem_stage = tmem_stage - num_tmem_stages
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 256 <= tx < 384:
            T.inc_max_nreg(224)
            c_epi1 = T.alloc_fragment((half_kv, 16), accum_dtype)
            logits_epi1 = T.alloc_fragment((half_kv,), accum_dtype)
            weights_epi1 = T.alloc_fragment((block_q, heads), accum_dtype)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            tmem_stage = T.alloc_var(T.int32, init=1)
            tmem_phase = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_epi1 in T.unroll(block_q - 1):
                    qi_scan_epi1 = qi_offset_epi1 + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_epi1])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_epi1])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                if num_kv_blocks > 0:
                    T.copy(weights_shared[q_stage, :, :], weights_epi1)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    kv_row = (first_bkv + kv_iter) * block_kv
                    tmem_col = tmem_stage * block_q * heads
                    T.assume(tmem_col >= 0)
                    T.assume(tmem_col + block_q * heads <= block_q * heads * num_tmem_stages)
                    T.mbarrier_wait_parity(tmem_full[tmem_stage], tmem_phase)
                    for qi_epi1 in T.unroll(block_q):
                        for bn_init_epi1 in T.Parallel(half_kv):
                            logits_epi1[bn_init_epi1] = T.float32(0)
                        for h_base_epi1 in T.unroll(heads // 16):
                            T.copy(
                                c_tmem[
                                    :,
                                    tmem_col
                                    + qi_epi1 * heads
                                    + h_base_epi1 * 16 : tmem_col
                                    + qi_epi1 * heads
                                    + (h_base_epi1 + 1) * 16,
                                ],
                                c_epi1,
                            )
                            for bn_reduce_epi1 in T.Parallel(half_kv):
                                for h_inner_epi1 in T.vectorized(16):
                                    h_epi1 = h_base_epi1 * 16 + h_inner_epi1
                                    logits_epi1[bn_reduce_epi1] += (
                                        T.max(
                                            c_epi1[bn_reduce_epi1, h_inner_epi1],
                                            T.float32(0),
                                        )
                                        * weights_epi1[qi_epi1, h_epi1]
                                    )
                        for bn_store_epi1 in T.Parallel(half_kv):
                            Logits[
                                q_row + qi_epi1, kv_row + half_kv + bn_store_epi1
                            ] = T.cast(logits_epi1[bn_store_epi1], logits_dtype)
                        T.sync_warp()
                    T.mbarrier_arrive(tmem_empty[tmem_stage])
                    tmem_stage = tmem_stage + tmem_stage_step
                    if tmem_stage >= num_tmem_stages:
                        tmem_phase = tmem_phase ^ 1
                        tmem_stage = tmem_stage - num_tmem_stages
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        T.sync_threads()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
)
def mqa_logits_fp8_persistent_ws_kernel(
    Q,
    KV,
    KVScale,
    Weights,
    KS,
    KE,
    Logits,
    seq_len: int,
    seq_len_kv: int,
    heads: int = 64,
    head_dim: int = 128,
    logits_stride: int = 4096,
    compressed_logits: bool = False,
    logits_dtype=T.float32,
):
    block_q = 128 // heads
    block_kv = 256
    half_kv = 128
    # The two KV halves are consecutive issues in a 2-phase TMEM stage ring.
    # Flip phase before wrapping stage: eager lowering guards each update separately.
    tmem_stage_step = block_kv // half_kv
    num_q_stages = 3
    num_stages = 3
    num_tmem_stages = 3
    accum_dtype = T.float32
    num_q_blocks = T.ceildiv(seq_len, block_q)
    sm_num = driver.get_num_sms()

    Q: T.Tensor((seq_len * heads, head_dim), T.float8_e4m3fn)
    KV: T.Tensor((seq_len_kv, head_dim), T.float8_e4m3fn)
    KVScale: T.Tensor((seq_len_kv,), accum_dtype)
    Weights: T.Tensor((seq_len, heads), accum_dtype)
    KS: T.Tensor((seq_len,), T.int32)
    KE: T.Tensor((seq_len,), T.int32)
    Logits: T.Tensor((seq_len, logits_stride), logits_dtype)

    with T.Kernel(sm_num, threads=384) as block_id:
        q_shared = T.alloc_shared((num_q_stages, block_q * heads, head_dim), T.float8_e4m3fn)
        weights_shared = T.alloc_shared((num_q_stages, block_q, heads), accum_dtype)
        kv_shared_0 = T.alloc_shared((num_stages, half_kv, head_dim), T.float8_e4m3fn)
        kv_shared_1 = T.alloc_shared((num_stages, half_kv, head_dim), T.float8_e4m3fn)
        kv_scale_shared = T.alloc_shared((num_stages, block_kv), accum_dtype)
        c_tmem = T.alloc_tmem((half_kv, block_q * heads * num_tmem_stages), accum_dtype)
        q_loaded = T.alloc_barrier([32] * num_q_stages)
        q_empty = T.alloc_barrier([320] * num_q_stages)
        kv_loaded = T.alloc_barrier([32] * num_stages)
        kv_empty = T.alloc_barrier([256] * num_stages)
        tmem_full = T.alloc_barrier([1] * num_tmem_stages)
        tmem_empty = T.alloc_barrier([128] * num_tmem_stages)

        tx = T.get_thread_binding()

        # Keep setmaxnreg inside the role branches. A separate pre-branch
        # setmaxnreg if emits the instruction but ptxas does not extend the
        # high-register region into the epilogue, and x16 TMEM loads spill.
        if tx < 32:
            T.dec_max_nreg(40)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            gemm_iter = T.alloc_var(T.int32, init=0)
            if q_block < num_q_blocks:
                first_q_row = q_block * block_q
                T.tma_copy(
                    Q[first_q_row * heads : first_q_row * heads + block_q * heads, :],
                    q_shared[0, :, :],
                    barrier=q_loaded[0],
                )
                T.tma_copy(
                    Weights[first_q_row : first_q_row + block_q, :],
                    weights_shared[0, :, :],
                    barrier=q_loaded[0],
                )
                T.mbarrier_arrive(q_loaded[0])
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_tail in T.unroll(block_q - 1):
                    qi_scan_tail = qi_offset_tail + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_tail])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_tail])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                next_q_block = q_block + sm_num
                next_q_iter = q_iter + 1
                next_q_stage = next_q_iter % num_q_stages
                next_q_phase = (next_q_iter // num_q_stages) & 1
                if next_q_block < num_q_blocks:
                    T.mbarrier_wait_parity(q_empty[next_q_stage], next_q_phase ^ 1)
                    next_q_row = next_q_block * block_q
                    T.tma_copy(
                        Q[next_q_row * heads : next_q_row * heads + block_q * heads, :],
                        q_shared[next_q_stage, :, :],
                        barrier=q_loaded[next_q_stage],
                    )
                    T.tma_copy(
                        Weights[next_q_row : next_q_row + block_q, :],
                        weights_shared[next_q_stage, :, :],
                        barrier=q_loaded[next_q_stage],
                    )
                    T.mbarrier_arrive(q_loaded[next_q_stage])

                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    kv_row = (first_bkv + kv_iter) * block_kv
                    stage = gemm_iter % num_stages
                    parity = (gemm_iter // num_stages) & 1
                    T.mbarrier_wait_parity(kv_empty[stage], parity ^ 1)
                    T.tma_copy(
                        KV[kv_row : kv_row + half_kv, :],
                        kv_shared_0[stage, :, :],
                        barrier=kv_loaded[stage],
                    )
                    T.tma_copy(
                        KV[kv_row + half_kv : kv_row + block_kv, :],
                        kv_shared_1[stage, :, :],
                        barrier=kv_loaded[stage],
                    )
                    T.tma_copy(KVScale[kv_row : kv_row + block_kv], kv_scale_shared[stage, :], barrier=kv_loaded[stage])
                    T.mbarrier_arrive(kv_loaded[stage])
                    gemm_iter = gemm_iter + 1
                    kv_iter = kv_iter + 1

                q_block = next_q_block
                q_iter = next_q_iter

        elif 32 <= tx < 64:
            T.dec_max_nreg(40)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            gemm_iter = T.alloc_var(T.int32, init=0)
            tmem_stage = T.alloc_var(T.int32, init=0)
            tmem_phase = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_tail in T.unroll(block_q - 1):
                    qi_scan_tail = qi_offset_tail + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_tail])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_tail])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    stage = gemm_iter % num_stages
                    parity = (gemm_iter // num_stages) & 1
                    tmem_col = tmem_stage * block_q * heads
                    T.mbarrier_wait_parity(tmem_empty[tmem_stage], tmem_phase ^ 1)
                    T.mbarrier_wait_parity(kv_loaded[stage], parity)
                    T.tcgen05_gemm(
                        kv_shared_0[stage, :, :],
                        q_shared[q_stage, :, :],
                        c_tmem[:, tmem_col : tmem_col + block_q * heads],
                        transpose_B=True,
                        mbar=tmem_full[tmem_stage],
                        clear_accum=True,
                    )
                    gemm_iter = gemm_iter + 1
                    tmem_stage = tmem_stage + tmem_stage_step
                    if tmem_stage >= num_tmem_stages:
                        tmem_phase = tmem_phase ^ 1
                        tmem_stage = tmem_stage - num_tmem_stages
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 64 <= tx < 96:
            T.dec_max_nreg(40)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            gemm_iter = T.alloc_var(T.int32, init=0)
            tmem_stage = T.alloc_var(T.int32, init=1)
            tmem_phase = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_tail1 in T.unroll(block_q - 1):
                    qi_scan_tail1 = qi_offset_tail1 + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_tail1])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_tail1])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    stage = gemm_iter % num_stages
                    parity = (gemm_iter // num_stages) & 1
                    tmem_col = tmem_stage * block_q * heads
                    T.mbarrier_wait_parity(tmem_empty[tmem_stage], tmem_phase ^ 1)
                    T.mbarrier_wait_parity(kv_loaded[stage], parity)
                    T.tcgen05_gemm(
                        kv_shared_1[stage, :, :],
                        q_shared[q_stage, :, :],
                        c_tmem[:, tmem_col : tmem_col + block_q * heads],
                        transpose_B=True,
                        mbar=tmem_full[tmem_stage],
                        clear_accum=True,
                    )
                    gemm_iter = gemm_iter + 1
                    tmem_stage = tmem_stage + tmem_stage_step
                    if tmem_stage >= num_tmem_stages:
                        tmem_phase = tmem_phase ^ 1
                        tmem_stage = tmem_stage - num_tmem_stages
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 96 <= tx < 128:
            T.dec_max_nreg(40)

        elif 128 <= tx < 256:
            T.inc_max_nreg(232)
            c_epi0 = T.alloc_fragment((half_kv, 16), accum_dtype)
            logits_epi0 = T.alloc_fragment((half_kv,), accum_dtype)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            gemm_iter = T.alloc_var(T.int32, init=0)
            tmem_stage = T.alloc_var(T.int32, init=0)
            tmem_phase = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_epi0 in T.unroll(block_q - 1):
                    qi_scan_epi0 = qi_offset_epi0 + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_epi0])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_epi0])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    kv_row = (first_bkv + kv_iter) * block_kv
                    stage = gemm_iter % num_stages
                    tmem_col = tmem_stage * block_q * heads
                    T.assume(tmem_col >= 0)
                    T.assume(tmem_col + block_q * heads <= block_q * heads * num_tmem_stages)
                    T.mbarrier_wait_parity(tmem_full[tmem_stage], tmem_phase)
                    for qi_epi0 in T.unroll(block_q):
                        for bn_init_epi0 in T.Parallel(half_kv):
                            logits_epi0[bn_init_epi0] = T.float32(0)
                        for h_base in T.unroll(heads // 16):
                            T.copy(
                                c_tmem[
                                    :,
                                    tmem_col + qi_epi0 * heads + h_base * 16 : tmem_col + qi_epi0 * heads + (h_base + 1) * 16,
                                ],
                                c_epi0,
                            )
                            for bn_reduce_epi0 in T.Parallel(half_kv):
                                for h_inner in T.vectorized(16):
                                    h = h_base * 16 + h_inner
                                    logits_epi0[bn_reduce_epi0] += (
                                        T.max(
                                            c_epi0[bn_reduce_epi0, h_inner],
                                            T.float32(0),
                                        )
                                        * weights_shared[q_stage, qi_epi0, h]
                                    )
                        for bn_store_epi0 in T.Parallel(half_kv):
                            Logits[q_row + qi_epi0, kv_row + bn_store_epi0] = T.cast(
                                logits_epi0[bn_store_epi0]
                                * kv_scale_shared[stage, bn_store_epi0],
                                logits_dtype,
                            )
                    T.mbarrier_arrive(tmem_empty[tmem_stage])
                    T.mbarrier_arrive(kv_empty[stage])
                    gemm_iter = gemm_iter + 1
                    tmem_stage = tmem_stage + tmem_stage_step
                    if tmem_stage >= num_tmem_stages:
                        tmem_phase = tmem_phase ^ 1
                        tmem_stage = tmem_stage - num_tmem_stages
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 256 <= tx < 384:
            T.inc_max_nreg(232)
            c_epi1 = T.alloc_fragment((half_kv, 16), accum_dtype)
            logits_epi1 = T.alloc_fragment((half_kv,), accum_dtype)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            gemm_iter = T.alloc_var(T.int32, init=0)
            tmem_stage = T.alloc_var(T.int32, init=1)
            tmem_phase = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_epi1 in T.unroll(block_q - 1):
                    qi_scan_epi1 = qi_offset_epi1 + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_epi1])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_epi1])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    kv_row = (first_bkv + kv_iter) * block_kv
                    stage = gemm_iter % num_stages
                    tmem_col = tmem_stage * block_q * heads
                    T.assume(tmem_col >= 0)
                    T.assume(tmem_col + block_q * heads <= block_q * heads * num_tmem_stages)
                    T.mbarrier_wait_parity(tmem_full[tmem_stage], tmem_phase)
                    for qi_epi1 in T.unroll(block_q):
                        for bn_init_epi1 in T.Parallel(half_kv):
                            logits_epi1[bn_init_epi1] = T.float32(0)
                        for h_base in T.unroll(heads // 16):
                            T.copy(
                                c_tmem[
                                    :,
                                    tmem_col + qi_epi1 * heads + h_base * 16 : tmem_col + qi_epi1 * heads + (h_base + 1) * 16,
                                ],
                                c_epi1,
                            )
                            for bn_reduce_epi1 in T.Parallel(half_kv):
                                for h_inner in T.vectorized(16):
                                    h = h_base * 16 + h_inner
                                    logits_epi1[bn_reduce_epi1] += (
                                        T.max(
                                            c_epi1[bn_reduce_epi1, h_inner],
                                            T.float32(0),
                                        )
                                        * weights_shared[q_stage, qi_epi1, h]
                                    )
                        for bn_store_epi1 in T.Parallel(half_kv):
                            Logits[
                                q_row + qi_epi1, kv_row + half_kv + bn_store_epi1
                            ] = T.cast(
                                logits_epi1[bn_store_epi1]
                                * kv_scale_shared[stage, half_kv + bn_store_epi1],
                                logits_dtype,
                            )
                    T.mbarrier_arrive(tmem_empty[tmem_stage])
                    T.mbarrier_arrive(kv_empty[stage])
                    gemm_iter = gemm_iter + 1
                    tmem_stage = tmem_stage + tmem_stage_step
                    if tmem_stage >= num_tmem_stages:
                        tmem_phase = tmem_phase ^ 1
                        tmem_stage = tmem_stage - num_tmem_stages
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        T.sync_threads()


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double().flatten()
    y = y.double().flatten()
    den = (x * x + y * y).sum()
    if den == 0:
        return 0.0
    return float((1 - 2 * (x * y).sum() / den).item())


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def pack_sf_u8_to_u32_1d(sf_u8: torch.Tensor) -> torch.Tensor:
    assert sf_u8.dtype == torch.uint8
    assert sf_u8.dim() == 2
    _, sf_k_padded = sf_u8.shape
    assert sf_k_padded % 4 == 0
    words = sf_u8.to(torch.int64)
    packed = (words[:, 0::4] | (words[:, 1::4] << 8) | (words[:, 2::4] << 16) | (words[:, 3::4] << 24)).to(torch.uint32)
    return packed.T.contiguous().reshape(-1)


_FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def fp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(_FP4_E2M1_VALUES, device=device, dtype=torch.float32)


def quantize_float_to_fp4_packed(x: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    assert k % 2 == 0
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries).reshape(m, k).to(torch.uint8)
    idx = idx | (((x < 0) & (idx != 0)).to(torch.uint8) << 3)
    lo = idx[:, 0::2]
    hi = idx[:, 1::2]
    return (lo | (hi << 4)).to(torch.int8)


def quantize_mxfp4_with_packed_ue8m0(x: torch.Tensor, gran_k: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert x.size(1) % 2 == 0
    mn, k = x.shape
    padded_k = int(T.align_up(k, gran_k))
    x_padded = torch.zeros((mn, padded_k), device=x.device, dtype=x.dtype)
    x_padded[:, :k] = x
    x_view = x_padded.view(mn, padded_k // gran_k, gran_k)
    amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = ceil_to_ue8m0(amax / 6.0)
    x_fp4 = quantize_float_to_fp4_packed((x_view * (1.0 / sf.unsqueeze(2))).reshape(mn, padded_k))[:, : k // 2].contiguous()
    sf_u8 = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    sf_k_padded = int(T.align_up(sf_u8.shape[1], 4))
    if sf_k_padded != sf_u8.shape[1]:
        sf_padded = torch.full((mn, sf_k_padded), 127, device=x.device, dtype=torch.uint8)
        sf_padded[:, : sf_u8.shape[1]] = sf_u8
    else:
        sf_padded = sf_u8
    return x_fp4, pack_sf_u8_to_u32_1d(sf_padded), sf_u8


def cast_back_from_mxfp4(x_fp4: torch.Tensor, sf_packed: torch.Tensor, logical_k: int, gran_k: int = 32) -> torch.Tensor:
    u = x_fp4.contiguous().view(torch.uint8)
    lut = fp4_lut(u.device)
    lo = lut[(u & 0x0F).long()]
    hi = lut[((u >> 4) & 0x0F).long()]
    x = torch.empty((u.shape[0], logical_k), device=u.device, dtype=torch.float32)
    x[:, 0::2] = lo[:, : logical_k // 2]
    x[:, 1::2] = hi[:, : logical_k // 2]
    sf_k_blocks = int(T.cdiv(logical_k, gran_k))
    sf_groups = int(T.cdiv(sf_k_blocks, 4))
    packed = sf_packed.view(sf_groups, u.shape[0]).T.contiguous().to(torch.int64)
    sf_u8 = torch.empty((u.shape[0], sf_groups * 4), device=u.device, dtype=torch.uint8)
    for i in range(4):
        sf_u8[:, i::4] = ((packed >> (8 * i)) & 0xFF).to(torch.uint8)
    scales = torch.pow(2.0, sf_u8[:, :sf_k_blocks].to(torch.float32) - 127.0)
    for bi in range(sf_k_blocks):
        k0 = bi * gran_k
        k1 = min(k0 + gran_k, logical_k)
        x[:, k0:k1] *= scales[:, bi : bi + 1]
    return x


def local_per_custom_dims_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().float().amax(dim=1).clamp_min(1e-4)
    scale = (amax / 448.0).contiguous()
    return (x.float() * (1.0 / scale[:, None])).to(torch.float8_e4m3fn).contiguous(), scale


def prepare_mqa_data(config: MQALogitsConfig, dtype: str):
    config.validate()
    torch.manual_seed(config.seed)
    q = torch.randn(config.seq_len, config.num_heads, config.head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(config.seq_len_kv, config.head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(config.seq_len, config.num_heads, device="cuda", dtype=torch.float32)
    ks, ke = generate_ks_ke(config)

    if dtype == "fp8":
        q_in = q.to(torch.float8_e4m3fn).contiguous()
        kv_in = local_per_custom_dims_cast_to_fp8(kv)
        q_sim = q_in.to(torch.bfloat16)
        kv_sim = (kv_in[0].float() * kv_in[1].unsqueeze(1)).to(torch.bfloat16)
        return {
            "q": q_sim,
            "kv": kv_sim,
            "q_in": q_in,
            "kv_in": kv_in,
            "weights": weights,
            "ks": ks,
            "ke": ke,
        }

    if dtype != "fp4":
        raise ValueError(f"unsupported dtype: {dtype}")
    if config.seq_len_kv % 256 != 0:
        raise ValueError("seq_len_kv must be divisible by 256 for the FP4 SOTA tile")

    q_fp4 = quantize_mxfp4_with_packed_ue8m0(q.view(-1, config.head_dim), gran_k=32)
    kv_fp4 = quantize_mxfp4_with_packed_ue8m0(kv.view(-1, config.head_dim), gran_k=32)
    q_sim = cast_back_from_mxfp4(q_fp4[0], q_fp4[1], config.head_dim, gran_k=32).view(config.seq_len, config.num_heads, config.head_dim)
    kv_sim = cast_back_from_mxfp4(kv_fp4[0], kv_fp4[1], config.head_dim, gran_k=32).view(config.seq_len_kv, config.head_dim)
    q_in = (
        q_fp4[0].view(config.seq_len, config.num_heads, config.head_dim // 2).contiguous(),
        q_fp4[1].view(config.seq_len, config.num_heads).contiguous(),
    )
    kv_in = (
        kv_fp4[0].view(config.seq_len_kv, config.head_dim // 2).contiguous(),
        kv_fp4[1].view(config.seq_len_kv).contiguous(),
    )
    return {
        "q": q_sim.to(torch.bfloat16),
        "kv": kv_sim.to(torch.bfloat16),
        "q_in": q_in,
        "kv_in": kv_in,
        "weights": weights,
        "ks": ks,
        "ke": ke,
    }


def run_fp8(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    logits_dtype: str = "float32",
) -> torch.Tensor:
    seq_len, heads, head_dim = q_fp8.shape
    seq_len_kv = kv_fp8.shape[0]
    MQALogitsConfig(seq_len, seq_len_kv, heads, head_dim, logits_dtype).validate()
    logits = torch.full(
        (seq_len, seq_len_kv),
        float("-inf"),
        device=q_fp8.device,
        dtype=_torch_logits_dtype(logits_dtype),
    )
    mqa_logits_fp8_persistent_ws_kernel(
        q_fp8.reshape(seq_len * heads, head_dim),
        kv_fp8,
        kv_scale,
        weights,
        ks,
        ke,
        logits,
        seq_len,
        seq_len_kv,
        heads=heads,
        head_dim=head_dim,
        logits_stride=seq_len_kv,
        compressed_logits=False,
        logits_dtype=_tilelang_logits_dtype(logits_dtype),
    )
    return logits


def run_fp4(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_fp4: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    logits_dtype: str = "float32",
) -> torch.Tensor:
    seq_len, heads, head_dim_packed = q_fp4.shape
    head_dim = head_dim_packed * 2
    seq_len_kv = kv_fp4.shape[0]
    MQALogitsConfig(seq_len, seq_len_kv, heads, head_dim, logits_dtype).validate()
    if seq_len_kv % 256 != 0:
        raise ValueError("seq_len_kv must be divisible by 256 for the FP4 SOTA tile")
    logits = torch.full(
        (seq_len, seq_len_kv),
        float("-inf"),
        device=q_fp4.device,
        dtype=_torch_logits_dtype(logits_dtype),
    )
    mqa_logits_fp4_persistent_ws_kernel(
        q_fp4.reshape(seq_len * heads, head_dim_packed),
        q_scale.reshape(-1),
        kv_fp4,
        kv_scale.reshape(-1),
        weights,
        ks,
        ke,
        logits,
        seq_len,
        seq_len_kv,
        heads=heads,
        head_dim=head_dim,
        logits_stride=seq_len_kv,
        compressed_logits=False,
        logits_dtype=_tilelang_logits_dtype(logits_dtype),
    )
    return logits


def run_example_case(config: MQALogitsConfig, dtype: str, check: bool = True) -> None:
    data = prepare_mqa_data(config, dtype)
    ref = ref_mqa_logits(data["q"], data["kv"], data["weights"], data["ks"], data["ke"])
    if dtype == "fp8":
        kv_fp8, kv_scale = data["kv_in"]
        out = run_fp8(data["q_in"], kv_fp8, kv_scale, data["weights"], data["ks"], data["ke"], config.logits_dtype)
    elif dtype == "fp4":
        q_fp4, q_scale = data["q_in"]
        kv_fp4, kv_scale = data["kv_in"]
        out = run_fp4(q_fp4, q_scale, kv_fp4, kv_scale, data["weights"], data["ks"], data["ke"], config.logits_dtype)
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    observed = out.float().masked_fill(ref == float("-inf"), 0)
    ref_cmp = ref.masked_fill(ref == float("-inf"), 0)
    diff = calc_diff(observed, ref_cmp)
    if check:
        threshold = 2e-3 if dtype == "fp4" else 1e-4
        assert diff < threshold, f"{dtype} diff {diff} >= {threshold}"
    print(f"{dtype} s{config.seq_len}_skv{config.seq_len_kv}_h{config.num_heads}_d{config.head_dim}_{config.logits_dtype}: diff={diff:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone SM100 MQA logits SOTA kernels.")
    parser.add_argument("--dtype", choices=("fp8", "fp4", "both"), default="both")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seq-len-kv", type=int, default=4096)
    parser.add_argument("--logits-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    cfg = MQALogitsConfig(
        seq_len=args.seq_len,
        seq_len_kv=args.seq_len_kv,
        logits_dtype=args.logits_dtype,
        seed=args.seed,
    )
    dtypes = ("fp8", "fp4") if args.dtype == "both" else (args.dtype,)
    for dtype in dtypes:
        run_example_case(cfg, dtype, check=not args.no_check)


if __name__ == "__main__":
    main()
