"""M/N/K MAD traversal: sibling K pipelines share a counter-versioned L0 ring."""

import tilelang.ascend.language as T

from ...config import Major
from ..common import MX_SF_DIVISOR, clamp, scale_region
from ..copy import make_l1_to_l0_copy


def make_compute(config, major_a, major_b, *, blockscaled=False):
    """Bind the operand layouts, optional MX scales, and MAD instruction at compile time."""
    MAD_M, MAD_N, MAD_K = config.mad_m, config.mad_n, config.mad_k
    assert config.block_k % MAD_K == 0
    if blockscaled:
        assert MAD_K % MX_SF_DIVISOR == 0

    num_mad_m_tiles = config.block_m // MAD_M
    num_mad_n_tiles = config.block_n // MAD_N
    sf_pairs_per_k_block = config.block_k // MX_SF_DIVISOR
    copy_a = make_l1_to_l0_copy(major_a)
    copy_b = make_l1_to_l0_copy(major_b)
    gemm = T.gemm_blockscaled if blockscaled else T.gemm

    @T.macro
    def compute(
        l1a,
        l1b,
        l0a,
        l0b,
        l0c,
        k_block_idx,
        num_k_blocks,
        actual_m,
        actual_n,
        actual_k,
        l1_sfa=None,
        l1_sfb=None,
        l0_sfa=None,
        l0_sfb=None,
        k_within_chunk=0,
    ):
        aligned_k = T.ceildiv(actual_k, MX_SF_DIVISOR) * MX_SF_DIVISOR if blockscaled else actual_k
        # Reuse only when each K owner advances the shared L0 counter by one full ring.
        reuse_l0 = T.ceildiv(aligned_k, MAD_K) == config.num_l0_stages
        # Keep the loading owner ahead of its reuse consumers after unrolling.
        with T.Stage(0):
            for m_mad_tile_idx in T.Unroll(num_mad_m_tiles, explicit=True):
                m_mad_idx = m_mad_tile_idx * MAD_M
                if m_mad_idx < actual_m:
                    eff_mad_m = clamp(actual_m - m_mad_idx, 0, MAD_M)
                    load_mad_m = T.ceildiv(eff_mad_m, 32) * 32 if blockscaled and major_a == Major.MN else eff_mad_m
                    for n_mad_tile_idx in T.Unroll(num_mad_n_tiles, explicit=True):
                        n_mad_idx = n_mad_tile_idx * MAD_N
                        if n_mad_idx < actual_n:
                            eff_mad_n = clamp(actual_n - n_mad_idx, 0, MAD_N)
                            load_mad_n = T.ceildiv(eff_mad_n, 32) * 32 if blockscaled and major_b == Major.MN else eff_mad_n
                            for k_mad_tile_idx in T.serial(
                                T.ceildiv(aligned_k, MAD_K),
                                annotations={
                                    # Retain K owners even when an operand is only read.
                                    "multi_buffer_eligible": [l0a, l0b, l0_sfa, l0_sfb] if blockscaled else [l0a, l0b]
                                },
                            ):
                                k_mad_idx = k_mad_tile_idx * MAD_K
                                eff_mad_k = clamp(aligned_k - k_mad_idx, 0, MAD_K)
                                sf_start = k_within_chunk * sf_pairs_per_k_block + k_mad_idx // MX_SF_DIVISOR
                                is_first = k_block_idx == 0 and k_mad_idx == 0
                                is_last = k_block_idx + 1 >= num_k_blocks and k_mad_idx + MAD_K >= aligned_k
                                unit_flag_ctrl = T.Select(is_last, 3, 2)
                                if n_mad_tile_idx == 0 or not reuse_l0:
                                    copy_a(
                                        l1a,
                                        l0a[0:load_mad_m, 0:eff_mad_k],
                                        m_mad_idx,
                                        k_mad_idx,
                                        load_mad_m,
                                        eff_mad_k,
                                        l1_scale=scale_region(l1_sfa, m_mad_idx, load_mad_m, sf_start, eff_mad_k),
                                        l0_scale=scale_region(l0_sfa, 0, load_mad_m, 0, eff_mad_k),
                                    )
                                if num_mad_n_tiles != 1 or m_mad_tile_idx == 0 or not reuse_l0:
                                    copy_b(
                                        l1b,
                                        l0b[0:load_mad_n, 0:eff_mad_k],
                                        n_mad_idx,
                                        k_mad_idx,
                                        load_mad_n,
                                        eff_mad_k,
                                        l1_scale=scale_region(l1_sfb, n_mad_idx, load_mad_n, sf_start, eff_mad_k),
                                        l0_scale=scale_region(l0_sfb, 0, load_mad_n, 0, eff_mad_k),
                                    )
                                scale_args = (
                                    {
                                        "SFA": scale_region(l0_sfa, 0, load_mad_m, 0, eff_mad_k),
                                        "SFB": scale_region(l0_sfb, 0, load_mad_n, 0, eff_mad_k),
                                    }
                                    if blockscaled
                                    else {}
                                )
                                gemm(
                                    l0a[0:load_mad_m, 0:eff_mad_k],
                                    l0b[0:load_mad_n, 0:eff_mad_k],
                                    l0c[m_mad_tile_idx, n_mad_tile_idx, 0:load_mad_m, 0:load_mad_n],
                                    transpose_B=True,
                                    clear_accum=is_first,
                                    unit_flag_ctrl=unit_flag_ctrl,
                                    **scale_args,
                                )

    return compute
