#pragma once

#include "kernel_operator.h"

namespace simd_inst {
template <typename T> struct vec {};

template <> struct vec<float> {
  using type = vector_f32;
};
template <> struct vec<half> {
  using type = vector_f16;
};

#if (__NPU_ARCH__ == 3510)
template <> struct vec<bfloat16_t> {
  using type = vector_bf16;
};
template <> struct vec<fp8_e4_t> {
  using type = vector_f8e4m3;
};
template <> struct vec<fp8_e5_t> {
  using type = vector_f8e5m2;
};
// template <> struct vec<float8_e4m3_t> {
//   using type = vector_f8e4m3;
// };
template <> struct vec<float8_e5m2_t> {
  using type = vector_f8e5m2;
};
template <> struct vec<float8_e8m0_t> {
  using type = vector_f8e8m0;
};
template <> struct vec<float4_e2m1x2_t> {
  using type = vector_f4e2m1x2;
};
template <> struct vec<float4_e1m2x2_t> {
  using type = vector_f4e1m2x2;
};
#endif

template <> struct vec<uint32_t> {
  using type = vector_u32;
};

template <> struct vec<uint16_t> {
  using type = vector_u16;
};

template <> struct vec<uint8_t> {
  using type = vector_u8;
};

template <> struct vec<int32_t> {
  using type = vector_s32;
};

template <> struct vec<int16_t> {
  using type = vector_s16;
};

template <> struct vec<int8_t> {
  using type = vector_s8;
};

template <> struct vec<int64_t> {
  using type = vector_s64;
};

template <> struct vec<uint64_t> {
  using type = vector_u64;
};

template <typename FirstVec, typename SecondVec = FirstVec> struct vec_pair {
  FirstVec v0;
  SecondVec v1;
};

template <typename T> using vec_t = typename vec<T>::type;

template <typename SrcVec> struct widen_vec {
  using type = SrcVec;
};

template <> struct widen_vec<vector_s8> {
  using type = vector_s16;
};

template <> struct widen_vec<vector_u8> {
  using type = vector_u16;
};

template <> struct widen_vec<vector_s16> {
  using type = vector_s32;
};

template <> struct widen_vec<vector_u16> {
  using type = vector_u32;
};

template <typename SrcVec> using widen_vec_t = typename widen_vec<SrcVec>::type;

template <typename T, typename Offset, typename Dist>
__simd_callee__ inline vec_t<T> vlds(__ubuf__ T *src, Offset offset,
                                     Dist dist) {
  vec_t<T> dst;
  ::vlds(dst, src, offset, dist);
  return dst;
}

template <typename Dist>
__simd_callee__ inline vector_bool plds(__ubuf__ uint32_t *src, int32_t offset,
                                        Dist dist) {
  vector_bool dst;
  ::plds(dst, src, offset, dist);
  return dst;
}

template <typename Dist>
__simd_callee__ inline void psts(vector_bool src, __ubuf__ uint32_t *base,
                                 int32_t offset, Dist dist) {
  ::psts(src, base, offset, dist);
}

// Dual-dest memory load (ASC DIST_DINTLV_B16). Prefer ::vld(dst0,dst1,...)
// which wraps ::vlds on dav-3510; matches asc_loadalign_v2_impl.h.
template <typename T, typename Dist>
__simd_callee__ inline vec_pair<vec_t<T>> vld_x2(__ubuf__ T *src, Dist dist) {
  vec_pair<vec_t<T>> dst;
  ::vld(dst.v0, dst.v1, src, dist);
  return dst;
}

template <typename T, typename Offset, typename Dist>
__simd_callee__ inline vec_pair<vec_t<T>> vld_x2(__ubuf__ T *src, Offset offset,
                                                 Dist dist) {
  vec_pair<vec_t<T>> dst;
  ::vld(dst.v0, dst.v1, src, offset, dist);
  return dst;
}

template <typename T>
__simd_callee__ inline vec_t<T> vgatherb(__ubuf__ T *base, vector_u32 idx) {
  vec_t<T> dst;
  ::vgatherb(dst, base, idx);
  return dst;
}

template <typename T>
__simd_callee__ inline vec_t<T> vgatherb(__ubuf__ T *base, vector_u32 idx,
                                         vector_bool mask) {
  vec_t<T> dst;
  ::vgatherb(dst, base, idx, mask);
  return dst;
}

template <typename T, typename IdxVec>
__simd_callee__ inline vec_t<T> vgather2(__ubuf__ T *base, IdxVec idx,
                                         vector_bool mask) {
  vec_t<T> dst;
  ::vgather2(dst, base, idx, mask);
  return dst;
}

template <typename IdxVec>
__simd_callee__ inline widen_vec_t<vec_t<int8_t>>
vgather2(__ubuf__ int8_t *base, IdxVec idx, vector_bool mask) {
  widen_vec_t<vec_t<int8_t>> dst;
  ::vgather2(dst, base, idx, mask);
  return dst;
}

template <typename IdxVec>
__simd_callee__ inline widen_vec_t<vec_t<uint8_t>>
vgather2(__ubuf__ uint8_t *base, IdxVec idx, vector_bool mask) {
  widen_vec_t<vec_t<uint8_t>> dst;
  ::vgather2(dst, base, idx, mask);
  return dst;
}

template <typename T, typename IdxVec>
__simd_callee__ inline void vscatter(vec_t<T> data, __ubuf__ T *base,
                                     IdxVec idx, vector_bool mask) {
  ::vscatter(data, base, idx, mask);
}

__simd_callee__ inline vector_bool pand(vector_bool src_0, vector_bool src_1,
                                        vector_bool mask) {
  vector_bool dst;
  ::pand(dst, src_0, src_1, mask);
  return dst;
}

__simd_callee__ inline vector_bool por(vector_bool src_0, vector_bool src_1,
                                       vector_bool mask) {
  vector_bool dst;
  ::por(dst, src_0, src_1, mask);
  return dst;
}

__simd_callee__ inline vector_bool pxor(vector_bool src_0, vector_bool src_1,
                                        vector_bool mask) {
  vector_bool dst;
  ::pxor(dst, src_0, src_1, mask);
  return dst;
}

__simd_callee__ inline vector_bool pnot(vector_bool src, vector_bool mask) {
  vector_bool dst;
  ::pnot(dst, src, mask);
  return dst;
}

__simd_callee__ inline vector_bool psel(vector_bool src_0, vector_bool src_1,
                                        vector_bool mask) {
  vector_bool dst;
  ::psel(dst, src_0, src_1, mask);
  return dst;
}

// f16->f32, f8->f32, i32->f32: ::vcvt(dst, src, mask, part/round, mode)
template <typename U, typename SrcVec, typename Part, typename Mode>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask, Part part,
                                     Mode mode) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, part, mode);
  return dst;
}

// f32->i32: ::vcvt(dst, src, mask, round, rs, mode)
template <typename U, typename SrcVec, typename RoundMode, typename Rs,
          typename Mode>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask,
                                     RoundMode roundingMode, Rs rsMode,
                                     Mode mode) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, roundingMode, rsMode, mode);
  return dst;
}

// f32->f16, f32->f8, f16->f8: ::vcvt(dst, src, mask, round, rs, part, mode)
template <typename U, typename SrcVec, typename RoundMode, typename Rs,
          typename Part, typename Mode>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask,
                                     RoundMode roundingMode, Rs rsMode,
                                     Part part, Mode mode) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, roundingMode, rsMode, part, mode);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<SrcVec> vdintlv(SrcVec src_0, SrcVec src_1) {
  vec_pair<SrcVec> dst;
  ::vdintlv(dst.v0, dst.v1, src_0, src_1);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<SrcVec> vintlv(SrcVec src_0, SrcVec src_1) {
  vec_pair<SrcVec> dst;
  ::vintlv(dst.v0, dst.v1, src_0, src_1);
  return dst;
}

// -- Binary arithmetic
// ----------------------------------------------------------

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vadd(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vadd(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<vector_bool, SrcVec>
vaddc(SrcVec src_0, SrcVec src_1, vector_bool mask) {
  vec_pair<vector_bool, SrcVec> dst;
  ::vaddc(dst.v0, dst.v1, src_0, src_1, mask);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<vector_bool, SrcVec>
vsubc(SrcVec src_0, SrcVec src_1, vector_bool mask) {
  vec_pair<vector_bool, SrcVec> dst;
  ::vsubc(dst.v0, dst.v1, src_0, src_1, mask);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<vector_bool, SrcVec>
vaddcs(SrcVec src_0, SrcVec src_1, vector_bool carrysrcp, vector_bool mask) {
  vec_pair<vector_bool, SrcVec> dst;
  ::vaddcs(dst.v0, dst.v1, src_0, src_1, carrysrcp, mask);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<vector_bool, SrcVec>
vsubcs(SrcVec src_0, SrcVec src_1, vector_bool carrysrcp, vector_bool mask) {
  vec_pair<vector_bool, SrcVec> dst;
  ::vsubcs(dst.v0, dst.v1, src_0, src_1, carrysrcp, mask);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<SrcVec, SrcVec>
vmull(SrcVec src_0, SrcVec src_1, vector_bool mask) {
  vec_pair<SrcVec, SrcVec> dst;
  ::vmull(dst.v0, dst.v1, src_0, src_1, mask);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsub(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vsub(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vmul(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vmul(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline void vmula(SrcVec *dst, SrcVec src_0, SrcVec src_1,
                                  vector_bool mask, Mode mode) {
  ::vmula(*dst, src_0, src_1, mask, mode);
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline void vmadd(SrcVec *dst, SrcVec src_0, SrcVec src_1,
                                  vector_bool mask, Mode mode) {
  ::vmadd(*dst, src_0, src_1, mask, mode);
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline void vaxpy(SrcVec *dst, SrcVec src, ScalarT scalar,
                                  vector_bool mask, Mode mode) {
  ::vaxpy(*dst, src, scalar, mask, mode);
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vdiv(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vdiv(dst, src_0, src_1, mask, mode);
  return dst;
}

// ============================================================================
// Precision division matching Ascend DivPrecisionImpl / torch.npu behavior.
// This intentionally keeps hardware FTZ/special-value behavior and only applies
// the 0ULP error-correction core.
// ============================================================================

// Precision f32 division exposed under the historical vdiv_0ulp_ftz_true name.
template <typename Mode>
__simd_callee__ inline vector_f32
vdiv_0ulp_ftz_true(vector_f32 src0, vector_f32 src1, vector_bool mask,
                   Mode mode) {
  constexpr uint32_t infNanBound = 0xff800000u;
  constexpr uint32_t signBitNum = 0x80000000u;

  vector_f32 regNegZero;
  ::vdup((vector_u32 &)regNegZero, signBitNum, mask, mode);

  vector_f32 z;
  ::vdiv(z, src0, src1, mask, mode);

  vector_u32 infNan;
  ::vor(infNan, (vector_u32 &)z, (vector_u32 &)regNegZero, mask, mode);

  vector_f32 tmpDst = z;

  vector_bool zeroCmp;
  ::vcmps_eq(zeroCmp, z, 0.0f, mask);
  vector_bool infNanCmp;
  ::vcmps_ge(infNanCmp, infNan, infNanBound, mask);
  ::por(infNanCmp, infNanCmp, zeroCmp, mask);

  vector_f32 y;
  ::vmuls(y, src1, -1.0f, mask, mode);
  vector_f32 r = src0;
  ::vmula(r, z, y, mask, mode);

  vector_f32 rPre, rNext, zPre, zNext;
  ::vadds((vector_s32 &)zPre, (vector_s32 &)z, -1, mask, mode);
  ::vadds((vector_s32 &)zNext, (vector_s32 &)z, 1, mask, mode);

  rPre = src0;
  rNext = src0;
  ::vmula(rPre, zPre, y, mask, mode);
  ::vmula(rNext, zNext, y, mask, mode);

  ::vabs(r, r, mask, mode);
  ::vabs(rPre, rPre, mask, mode);
  ::vabs(rNext, rNext, mask, mode);

  vector_bool cmpMaskReg;
  ::vcmp_lt(cmpMaskReg, r, rPre, mask);
  ::vsel(r, r, rPre, cmpMaskReg);
  ::vsel(z, z, zPre, cmpMaskReg);

  ::vcmp_lt(cmpMaskReg, rNext, r, mask);
  ::vsel(z, zNext, z, cmpMaskReg);

  vector_f32 dst;
  ::vsel(dst, tmpDst, z, infNanCmp);
  return dst;
}

template <typename Mode>
__simd_callee__ inline void vdiv_0ulp_ftz_true(vector_f32 &dst, vector_f32 src0,
                                               vector_f32 src1,
                                               vector_bool mask, Mode mode) {
  (void)mode;
  // The in-place overload implements MODE_MERGING: update active lanes while
  // preserving the old destination value for inactive lanes.
  vector_f32 result = vdiv_0ulp_ftz_true(src0, src1, mask, MODE_ZEROING);
  ::vsel(dst, result, dst, mask);
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vmax(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vmax(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vmin(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vmin(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vand(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vand(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vor(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                  Mode mode) {
  SrcVec dst;
  ::vor(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vxor(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vxor(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename ShiftVec, typename Mode>
__simd_callee__ inline SrcVec vshl(SrcVec src_0, ShiftVec src_1,
                                   vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vshl(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename ShiftVec, typename Mode>
__simd_callee__ inline SrcVec vshr(SrcVec src_0, ShiftVec src_1,
                                   vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vshr(dst, src_0, src_1, mask, mode);
  return dst;
}

// -- Unary
// ----------------------------------------------------------------------

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vln(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vln(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsqrt(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vsqrt(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vabs(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vabs(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vneg(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vneg(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vrelu(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vrelu(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT>
__simd_callee__ inline SrcVec vlrelu(SrcVec src, ScalarT alpha,
                                     vector_bool mask) {
  SrcVec dst;
  ::vlrelu(dst, src, alpha, mask, MODE_ZEROING);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline SrcVec vprelu(SrcVec src_0, SrcVec src_1,
                                     vector_bool mask) {
  SrcVec dst;
  ::vprelu(dst, src_0, src_1, mask, MODE_UNKNOWN);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vnot(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vnot(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vexp(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vexp(dst, src, mask, mode);
  return dst;
}

// ============================================================================
// SFU precision wrappers.  Naming convention: <op>_<N>ulp_ftz_<mode> maps 1:1
// to the CANN PRECISION_<N>ULP_FTZ_<mode> algorithm tiers
// (kernel_reg_compute_utils.h):
//   vdiv_0ulp_ftz_true    = DivAlgo::PRECISION_0ULP_FTZ_TRUE (DivPrecisionImpl)
//   vexp_1ulp_ftz_false   = ExpAlgo::PRECISION_1ULP_FTZ_FALSE  (ExpPrecision)
//   vln_1ulp_ftz_false    = LnAlgo::PRECISION_1ULP_FTZ_FALSE
//   vsqrt_0ulp_ftz_false  = SqrtAlgo::PRECISION_0ULP_FTZ_FALSE
//   (SqrtFastInverseImpl)
// FTZ_FALSE wrappers preserve subnormal inputs/outputs that the hardware SFU
// flushes to zero. Selected per op via the `precision='ftz_false'`
// kwarg (see codegen_ascend.cc). Structurally identical to the CANN 9.1.0
// precision sub-paths (ExpPrecision / ln / SqrtPrecision scale-unscale);
// these are full-vector costs applied to every lane, so keep the default off.
// ============================================================================

// vexp FTZ_FALSE: normal outputs pass through; outputs that would land in the
// subnormal range are computed as (e^(x/2))^2 so the SFU input stays normal.
// Self-consistent at x < -174.7: e^(x/2) itself is flushed to 0, squaring
// yields 0, and the true e^x < 2^-252 correctly rounds to 0.
template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vexp_1ulp_ftz_false(SrcVec src, vector_bool mask,
                                                  Mode mode) {
  constexpr float kMaxSubnormal =
      1.1754942e-38f; // largest subnormal (2^-126 - 2^-149)
  SrcVec z, t, half;
  vector_bool m;
  ::vexp(z, src, mask, mode);            // SFU initial value
  ::vcmps_le(m, z, kMaxSubnormal, mask); // output would be subnormal?
  ::vmuls(half, src, 0.5f, mask, mode);
  ::vexp(t, half, mask, mode); // e^(x/2): stays in normal range
  ::vmul(t, t, t, mask, mode);
  SrcVec dst;
  ::vsel(dst, t, z, m);
  return dst;
}

// In-place overload implementing MODE_MERGING (mirrors vdiv_0ulp_ftz_true):
// active lanes get the FTZ_FALSE result, inactive lanes keep their old value.
template <typename SrcVec, typename Mode>
__simd_callee__ inline void vexp_1ulp_ftz_false(SrcVec &dst, SrcVec src,
                                                vector_bool mask, Mode mode) {
  (void)mode;
  SrcVec result = vexp_1ulp_ftz_false(src, mask, MODE_ZEROING);
  ::vsel(dst, result, dst, mask);
}

// vln FTZ_FALSE: positive subnormal inputs are scaled by 2^23 before VLN and
// compensated by -ln(2^23) (the scaled value is exact: subnormal mantissa
// shifted into the normal range). Other inputs keep hardware semantics
// (0 -> -inf, negative -> NaN).
template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vln_1ulp_ftz_false(SrcVec src, vector_bool mask,
                                                 Mode mode) {
  // 1.1754944e-38f rounds to 2^-126 (smallest normal), so the
  // `src < kMinNormal` test below is exactly the subnormal-input test.
  constexpr float kMinNormal = 1.1754944e-38f;   // 2^-126, smallest normal
  constexpr float kLn2p23 = 15.942385152878742f; // ln(2^23)
  SrcVec z, t, scaled;
  vector_bool sub, pos, m;
  ::vln(z, src, mask, mode);              // hardware path
  ::vcmps_lt(sub, src, kMinNormal, mask); // subnormal magnitude
  ::vcmps_gt(pos, src, 0.0f, mask);       // positive only
  ::pand(m, sub, pos, mask);
  ::vmuls(scaled, src, 8388608.0f, mask, mode); // 2^23
  ::vln(t, scaled, mask, mode);
  ::vadds(t, t, -kLn2p23, mask, mode);
  SrcVec dst;
  ::vsel(dst, t, z, m);
  return dst;
}

// In-place overload implementing MODE_MERGING (mirrors vdiv_0ulp_ftz_true).
template <typename SrcVec, typename Mode>
__simd_callee__ inline void vln_1ulp_ftz_false(SrcVec &dst, SrcVec src,
                                               vector_bool mask, Mode mode) {
  (void)mode;
  SrcVec result = vln_1ulp_ftz_false(src, mask, MODE_ZEROING);
  ::vsel(dst, result, dst, mask);
}

// vsqrt FTZ_FALSE: replica of CANN 9.1.0 SqrtFastInverseImpl
// (PRECISION_0ULP_FTZ_FALSE / FAST_INVERSE), chosen over the 1ULP_FTZ_FALSE
// scale/unscale variant because the latter mis-rounds 0x007fffff to +0.
// Inputs < 1 are scaled by 2^24 so the chain stays in the normal range,
// then unscaled by 2^-12; a 1/sqrt initial value plus Newton and a second
// residual correction gives correct rounding; +-0 and +inf pass through.
template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsqrt_0ulp_ftz_false(SrcVec src, vector_bool mask,
                                                   Mode mode) {
  constexpr float kOne = 1.0f;
  constexpr float kHalf = 0.5f;
  constexpr float kScaleUp = 16777216.0f;     // 2^24
  constexpr float kScaleDn = 0.000244140625f; // 2^-12
  constexpr float kPosInf = __builtin_inff(); // true +infinity
  SrcVec b, scaled, one, tmp, err, res, x;
  vector_bool p, isZero, isInf, special;
  ::vcmps_lt(p, src, kOne, mask); // scale inputs < 1 up
  ::vmuls(scaled, src, kScaleUp, mask, mode);
  ::vsel(b, scaled, src, p);
  ::vdup(one, kOne, mask, mode);
  ::vsqrt(tmp, b, mask, mode);
  ::vdiv(x, one, tmp, mask, mode); // 1/sqrt(b) initial value
  ::vmuls(tmp, x, -kOne, mask, mode);
  ::vmul(err, x, b, mask, mode);
  ::vmula(one, err, tmp, mask, mode); // first Newton step
  ::vmuls(tmp, x, kHalf, mask, mode);
  ::vmula(x, one, tmp, mask, mode);
  ::vmul(res, x, b, mask, mode);
  ::vmuls(tmp, res, -kOne, mask, mode);
  err = b;
  ::vmula(err, res, tmp, mask, mode); // second residual correction
  ::vmuls(tmp, x, kHalf, mask, mode);
  ::vmadd(tmp, err, res, mask, mode);
  ::vmuls(scaled, tmp, kScaleDn, mask, mode);
  ::vsel(tmp, scaled, tmp, p);           // unscale only the scaled inputs
  ::vcmps_eq(isZero, src, 0.0f, mask);   // +-0 pass through
  ::vcmps_eq(isInf, src, kPosInf, mask); // +inf pass through
  ::por(special, isZero, isInf, mask);
  SrcVec dst;
  ::vsel(dst, src, tmp, special); // special ? src : tmp
  return dst;
}

// In-place overload implementing MODE_MERGING (mirrors vdiv_0ulp_ftz_true).
template <typename SrcVec, typename Mode>
__simd_callee__ inline void vsqrt_0ulp_ftz_false(SrcVec &dst, SrcVec src,
                                                 vector_bool mask, Mode mode) {
  (void)mode;
  SrcVec result = vsqrt_0ulp_ftz_false(src, mask, MODE_ZEROING);
  ::vsel(dst, result, dst, mask);
}

// -- Cross-lane reductions
// ------------------------------------------------------
template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcpadd(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcpadd(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline widen_vec_t<SrcVec> vcadd(SrcVec src, vector_bool mask,
                                                 Mode mode) {
  widen_vec_t<SrcVec> dst;
  ::vcadd(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcmax(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcmax(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcmin(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcmin(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcgadd(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcgadd(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcgmax(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcgmax(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcgmin(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcgmin(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsqz(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vsqz(dst, src, mask, mode);
  return dst;
}

// vusqz: per-lane exclusive prefix count of mask; merge-mode needs pre-zeroed
// Vd via vdup (a `{}` init lowers to a BUILD_VECTOR the backend rejects).
template <typename T> __simd_callee__ inline vec_t<T> vusqz(vector_bool mask) {
  vec_t<T> dst;
  ::vdup(dst, static_cast<T>(0), mask, MODE_ZEROING);
  ::vusqz(dst, mask);
  return dst;
}

__simd_callee__ inline vector_bool update_mask_b8(uint32_t value) {
  uint32_t v = value;
  return ::plt_b8(v, POST_UPDATE);
}

__simd_callee__ inline vector_bool update_mask_b16(uint32_t value) {
  uint32_t v = value;
  return ::plt_b16(v, POST_UPDATE);
}

__simd_callee__ inline vector_bool update_mask_b32(uint32_t value) {
  uint32_t v = value;
  return ::plt_b32(v, POST_UPDATE);
}

template <typename Part>
__simd_callee__ inline vector_bool ppack(vector_bool src, Part part) {
  vector_bool dst;
  ::ppack(dst, src, part);
  return dst;
}

template <typename Part>
__simd_callee__ inline vector_bool punpack(vector_bool src, Part part) {
  vector_bool dst;
  ::punpack(dst, src, part);
  return dst;
}

#define TL_SIMD_PINTLV_IMPL(WIDTH)                                             \
  __simd_callee__ inline vec_pair<vector_bool> pintlv_##WIDTH(                 \
      vector_bool src_0, vector_bool src_1) {                                  \
    vec_pair<vector_bool> dst;                                                 \
    ::pintlv_##WIDTH(dst.v0, dst.v1, src_0, src_1);                            \
    return dst;                                                                \
  }                                                                            \
  __simd_callee__ inline vec_pair<vector_bool> pdintlv_##WIDTH(                \
      vector_bool src_0, vector_bool src_1) {                                  \
    vec_pair<vector_bool> dst;                                                 \
    ::pdintlv_##WIDTH(dst.v0, dst.v1, src_0, src_1);                           \
    return dst;                                                                \
  }

TL_SIMD_PINTLV_IMPL(b8)
TL_SIMD_PINTLV_IMPL(b16)
TL_SIMD_PINTLV_IMPL(b32)
#undef TL_SIMD_PINTLV_IMPL

template <typename Bin>
__simd_callee__ inline void dhistv2(vector_u16 *dst, vector_u8 src,
                                    vector_bool mask, Bin bin) {
  ::dhistv2(*dst, src, mask, bin);
}

template <typename Bin>
__simd_callee__ inline void chistv2(vector_u16 *dst, vector_u8 src,
                                    vector_bool mask, Bin bin) {
  ::chistv2(*dst, src, mask, bin);
}

// -- Index ramp / compare
// ------------------------------------------------------------

// index ramp: T = int32_t / float / ...; returns dst[lane] = index (+/-) lane
template <typename T, typename Order>
__simd_callee__ inline vec_t<T> vci(T index, Order order) {
  vec_t<T> dst;
  ::vci(dst, index, order);
  return dst;
}

#define __SIMD_INST_VCMP(OP)                                                   \
  template <typename SrcVec>                                                   \
  __simd_callee__ inline vector_bool vcmp_##OP(SrcVec src_0, SrcVec src_1,     \
                                               vector_bool mask) {             \
    vector_bool dst;                                                           \
    ::vcmp_##OP(dst, src_0, src_1, mask);                                      \
    return dst;                                                                \
  }                                                                            \
  template <typename SrcVec, typename ScalarT>                                 \
  __simd_callee__ inline vector_bool vcmps_##OP(SrcVec src, ScalarT scalar,    \
                                                vector_bool mask) {            \
    vector_bool dst;                                                           \
    ::vcmps_##OP(dst, src, scalar, mask);                                      \
    return dst;                                                                \
  }
__SIMD_INST_VCMP(eq)
__SIMD_INST_VCMP(ne)
__SIMD_INST_VCMP(gt)
__SIMD_INST_VCMP(ge)
__SIMD_INST_VCMP(lt)
__SIMD_INST_VCMP(le)
#undef __SIMD_INST_VCMP

// -- Broadcast
// ------------------------------------------------------------------

// scalar broadcast: T = float / half / int32_t / ...
template <typename T, typename Mode>
__simd_callee__ inline vec_t<T> vdup(T src, vector_bool mask, Mode mode) {
  vec_t<T> dst;
  ::vdup(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Pos, typename Mode>
__simd_callee__ inline SrcVec vdupv(SrcVec src, vector_bool mask, Pos pos,
                                    Mode mode) {
  SrcVec dst;
  ::vdup(dst, src, mask, pos, mode);
  return dst;
}

// -- Select
// ---------------------------------------------------------------------

template <typename SrcVec>
__simd_callee__ inline SrcVec vsel(SrcVec src_0, SrcVec src_1,
                                   vector_bool mask) {
  SrcVec dst;
  ::vsel(dst, src_0, src_1, mask);
  return dst;
}

template <typename SrcVec, typename IdxVec>
__simd_callee__ inline SrcVec vselr(SrcVec src, IdxVec idx) {
  SrcVec dst;
  ::vselr(dst, src, idx);
  return dst;
}

// -- Scalar-vector ops
// ----------------------------------------------------------

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vadds(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vadds(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vmaxs(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vmaxs(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vmins(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vmins(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vmuls(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vmuls(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vshls(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vshls(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vshrs(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vshrs(dst, src, scalar, mask, mode);
  return dst;
}

// -- Exponential difference
// -----------------------------------------------------

template <typename SrcVec, typename Part>
__simd_callee__ inline SrcVec vexpdif(SrcVec src_0, SrcVec src_1,
                                      vector_bool mask, Part part) {
  SrcVec dst;
  ::vexpdif(dst, src_0, src_1, mask, part);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vabsdif(SrcVec src_0, SrcVec src_1,
                                      vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vabsdif(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename U, typename SrcVec, typename Part>
__simd_callee__ inline vec_t<U> vpack(SrcVec src, Part part) {
  vec_t<U> dst;
  ::vpack(dst, src, part);
  return dst;
}

template <typename SrcVec, typename Part>
__simd_callee__ inline widen_vec_t<SrcVec> vunpack(SrcVec src, Part part) {
  widen_vec_t<SrcVec> dst;
  ::vunpack(dst, src, part);
  return dst;
}

template <typename T>
__simd_callee__ inline void vsstb(vec_t<T> src, __ubuf__ T *base,
                                  int32_t stride, vector_bool mask) {
  ::vsstb(src, base, stride, mask);
}

template <typename T, typename Post>
__simd_callee__ inline __ubuf__ T *vsstb(vec_t<T> src, __ubuf__ T *base,
                                         int32_t stride, vector_bool mask,
                                         Post post) {
  ::vsstb(src, base, stride, mask, post);
  return base;
}

template <typename T, typename Dist>
__simd_callee__ inline void vsts(vec_t<T> data, __ubuf__ T *base,
                                 int32_t offset, Dist dist, vector_bool mask) {
  ::vsts(data, base, offset, dist, mask);
}

template <typename T> __simd_callee__ inline void mem_bar(T mem_type) {
  ::mem_bar(mem_type);
}

} // namespace simd_inst
