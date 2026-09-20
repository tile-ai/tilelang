from __future__ import annotations

from ptodsl import pto


def vdiv_precise_f32(src0, src1, mask):
    """Emit precise f32 vector division matching Ascend ``vdiv_precise``."""
    div_z0 = pto.vdiv(src0, src1, mask)
    div_z_bits = pto.vbitcast(div_z0, pto.ui32)
    div_special_bits = pto.vor(
        div_z_bits,
        pto.vbr(pto.ui32(0x80000000)),
        mask,
    )
    div_zero_cmp = pto.vcmps(div_z0, 0.0, mask, pto.CmpMode.EQ)
    div_inf_nan_cmp = pto.vcmp(
        div_special_bits,
        pto.vbr(pto.ui32(0xFF800000)),
        mask,
        pto.CmpMode.GE,
    )
    div_special_cmp = pto.por(div_inf_nan_cmp, div_zero_cmp, mask)

    div_neg_rhs = pto.vmuls(src1, -1.0, mask)
    div_r0 = pto.vmula(src0, div_z0, div_neg_rhs, mask)
    div_z_i32 = pto.vbitcast(div_z0, pto.i32)
    div_z_pre = pto.vbitcast(pto.vadds(div_z_i32, -1, mask), pto.f32)
    div_z_next = pto.vbitcast(pto.vadds(div_z_i32, 1, mask), pto.f32)
    div_r_pre = pto.vmula(src0, div_z_pre, div_neg_rhs, mask)
    div_r_next = pto.vmula(src0, div_z_next, div_neg_rhs, mask)

    div_r0_abs = pto.vabs(div_r0, mask)
    div_r_pre_abs = pto.vabs(div_r_pre, mask)
    div_r_next_abs = pto.vabs(div_r_next, mask)
    div_keep_current = pto.vcmp(
        div_r0_abs,
        div_r_pre_abs,
        mask,
        pto.CmpMode.LT,
    )
    div_best_r = pto.vsel(div_r0_abs, div_r_pre_abs, div_keep_current)
    div_best_z = pto.vsel(div_z0, div_z_pre, div_keep_current)
    div_better_next = pto.vcmp(
        div_r_next_abs,
        div_best_r,
        mask,
        pto.CmpMode.LT,
    )
    div_corrected = pto.vsel(div_z_next, div_best_z, div_better_next)
    return pto.vsel(div_z0, div_corrected, div_special_cmp)
