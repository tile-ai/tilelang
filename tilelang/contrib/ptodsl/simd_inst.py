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


def vexp_1ulp_ftz_false(src, mask):
    """Emit subnormal-preserving f32 vexp matching Ascend ``vexp_1ulp_ftz_false``.

    Normal outputs pass through; outputs that would land in the subnormal
    range are computed as ``(e^(x/2))^2`` so the SFU input stays normal.
    Self-consistent at x < -174.7: ``e^(x/2)`` itself is flushed to 0,
    squaring yields 0, and the true ``e^x < 2^-252`` correctly rounds to 0.
    """
    exp_z = pto.vexp(src, mask)
    exp_sub = pto.vcmps(exp_z, 1.1754942e-38, mask, pto.CmpMode.LE)
    exp_half = pto.vmuls(src, 0.5, mask)
    exp_t = pto.vexp(exp_half, mask)
    exp_t = pto.vmul(exp_t, exp_t, mask)
    return pto.vsel(exp_t, exp_z, exp_sub)


def vln_1ulp_ftz_false(src, mask):
    """Emit subnormal-preserving f32 vln matching Ascend ``vln_1ulp_ftz_false``.

    Positive subnormal inputs are scaled by ``2^23`` before VLN and
    compensated by ``-ln(2^23)`` (the scaled value is exact: the subnormal
    mantissa shifts into the normal range). Other inputs keep hardware
    semantics (0 -> -inf, negative -> NaN).
    """
    ln_z = pto.vln(src, mask)
    ln_sub = pto.vcmps(src, 1.1754944e-38, mask, pto.CmpMode.LT)
    ln_pos = pto.vcmps(src, 0.0, mask, pto.CmpMode.GT)
    ln_m = pto.pand(ln_sub, ln_pos, mask)
    ln_scaled = pto.vmuls(src, 8388608.0, mask)
    ln_t = pto.vln(ln_scaled, mask)
    ln_t = pto.vadds(ln_t, -15.942385152878742, mask)
    return pto.vsel(ln_t, ln_z, ln_m)


def vsqrt_0ulp_ftz_false(src, mask):
    """Emit subnormal-preserving f32 vsqrt matching Ascend ``vsqrt_0ulp_ftz_false``.

    Replica of CANN ``SqrtFastInverseImpl`` (``PRECISION_0ULP_FTZ_FALSE`` /
    ``FAST_INVERSE``), chosen over the 1ULP scale/unscale variant because the
    latter mis-rounds ``0x007fffff`` to +0. Inputs < 1 are scaled by ``2^24``
    so the chain stays in the normal range, then unscaled by ``2^-12``; a
    1/sqrt initial value plus a Newton step and a second residual correction
    gives correct rounding; +-0 and +inf pass through.
    """
    sqrt_p = pto.vcmps(src, 1.0, mask, pto.CmpMode.LT)
    sqrt_scaled = pto.vmuls(src, 16777216.0, mask)
    sqrt_b = pto.vsel(sqrt_scaled, src, sqrt_p)
    sqrt_one = pto.vbr(pto.f32(1.0))
    sqrt_tmp = pto.vsqrt(sqrt_b, mask)
    sqrt_x = pto.vdiv(sqrt_one, sqrt_tmp, mask)
    sqrt_tmp = pto.vmuls(sqrt_x, -1.0, mask)
    sqrt_err = pto.vmul(sqrt_x, sqrt_b, mask)
    sqrt_one = pto.vmula(sqrt_one, sqrt_err, sqrt_tmp, mask)
    sqrt_tmp = pto.vmuls(sqrt_x, 0.5, mask)
    sqrt_x = pto.vmula(sqrt_x, sqrt_one, sqrt_tmp, mask)
    sqrt_res = pto.vmul(sqrt_x, sqrt_b, mask)
    sqrt_tmp = pto.vmuls(sqrt_res, -1.0, mask)
    sqrt_err = pto.vmula(sqrt_b, sqrt_res, sqrt_tmp, mask)
    sqrt_tmp = pto.vmuls(sqrt_x, 0.5, mask)
    sqrt_tmp = pto.vmadd(sqrt_tmp, sqrt_err, sqrt_res, mask)
    sqrt_scaled = pto.vmuls(sqrt_tmp, 0.000244140625, mask)
    sqrt_tmp = pto.vsel(sqrt_scaled, sqrt_tmp, sqrt_p)
    sqrt_is_zero = pto.vcmps(src, 0.0, mask, pto.CmpMode.EQ)
    sqrt_is_inf = pto.vcmps(src, float("inf"), mask, pto.CmpMode.EQ)
    sqrt_special = pto.por(sqrt_is_zero, sqrt_is_inf, mask)
    return pto.vsel(src, sqrt_tmp, sqrt_special)
