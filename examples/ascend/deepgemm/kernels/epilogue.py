"""Output tiling and AIV transforms shared by the BF16 and FP8 kernel families."""

import tilelang.ascend.language as T

from ..config import Major
from .common import clamp, matrix_region


@T.macro
def apply_alpha(values, output, alpha):
    """Scale in FP32, then pack BF16 in place within each MAD tile's UB slot."""
    with T.SimdVF():
        # Expose the linear offset directly; equivalent row/column indexing can
        # leave division and remainder in generated code (e.g. MAD_N=48).
        flat_values = T.reshape(values, (values.shape[0] * values.shape[1],))
        flat_output = T.reshape(output, (output.shape[0] * output.shape[1],))
        mask = T.simd.pset(32)
        for i in T.serial(values.shape[0] * values.shape[1] // 64):
            value = T.simd.vld(flat_values[i * 64])
            scaled = T.simd.vmuls(value, alpha, mask)
            if output.dtype == "bfloat16":
                converted = T.simd.vcvt(scaled, "bfloat16")
                T.simd.vsts(flat_output[i * 64], converted, mask, dist="PK_B32")
            else:
                T.simd.vsts(flat_output[i * 64], scaled, mask)


def make_store_output(config, major_a, major_b, *, blockscaled=False, with_alpha=False):
    """Drain all MAD tiles in a block; GM stores exclude FP8 load/fixpipe padding."""
    MAD_M, MAD_N = config.mad_m, config.mad_n
    num_mad_m_tiles = config.block_m // MAD_M
    num_mad_n_tiles = config.block_n // MAD_N

    @T.macro
    def store_output(
        l0c,
        epilogue_ub,
        output_ub,
        gm_d,
        m_idx,
        n_idx,
        actual_m,
        actual_n,
        alpha=1.0,
        batch_idx=0,
    ):
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
                        # AutoSchedule assigns ownership per loop body; isolate each MAD tile,
                        # including both views of the in-place UB storage.
                        for _ in range(1):
                            T.assume_no_conflict(gm_d)
                            T.copy(
                                l0c[m_mad_tile_idx, n_mad_tile_idx, 0:load_mad_m, 0:load_mad_n],
                                epilogue_ub[0:load_mad_m, 0:load_mad_n],
                                unit_flag_ctrl=3,
                            )
                            if with_alpha:
                                apply_alpha(epilogue_ub, output_ub, alpha)
                            T.copy(
                                output_ub[0:eff_mad_m, 0:eff_mad_n],
                                matrix_region(gm_d, m_idx + m_mad_idx, n_idx + n_mad_idx, eff_mad_m, eff_mad_n, batch_idx),
                                l2_cache_ctrl=config.l2_ctrl_store_cd,
                            )

    return store_output
