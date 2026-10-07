#pragma once

#include <type_traits>

#include "c_api/asc_simd.h"

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

// C APIs produce a zeroing result for masked register operations. Legalized
// MODE_MERGING calls use this helper to preserve inactive destination lanes,
// matching the software merge in the CCE intrinsics on dav-3510.
template <typename Vec>
__simd_callee__ inline Vec merge_masked(Vec &dst, Vec result,
                                        vector_bool mask) {
  Vec old_dst = dst;
  asc_select(dst, result, old_dst, mask);
  return dst;
}

template <typename Vec, typename Pattern>
__simd_callee__ inline vector_bool reduction_result_mask(Pattern pattern) {
  if constexpr (std::is_same<Vec, vector_s8>::value ||
                std::is_same<Vec, vector_u8>::value) {
    return asc_create_mask_b8(pattern);
  } else if constexpr (std::is_same<Vec, vector_s16>::value ||
                       std::is_same<Vec, vector_u16>::value ||
                       std::is_same<Vec, vector_f16>::value ||
                       std::is_same<Vec, vector_bf16>::value) {
    return asc_create_mask_b16(pattern);
  } else {
    return asc_create_mask_b32(pattern);
  }
}

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

// Single-distribution register loads. T determines the element width;
// codegen selects one C API per distribution family.
#define SIMD_INST_DEFINE_LOAD(Op, CApi)                                        \
  template <typename T, typename Offset>                                       \
  __simd_callee__ inline vec_t<T> Op(__ubuf__ T *src, Offset offset) {         \
    vec_t<T> dst;                                                              \
    CApi(dst, src, offset);                                                    \
    return dst;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  __simd_callee__ inline vec_pair<vec_t<T>, __ubuf__ T *> Op##_postupdate(     \
      __ubuf__ T *src, int32_t offset) {                                       \
    vec_pair<vec_t<T>, __ubuf__ T *> dst;                                      \
    CApi##_postupdate(dst.v0, src, offset);                                    \
    dst.v1 = src;                                                              \
    return dst;                                                                \
  }

SIMD_INST_DEFINE_LOAD(vlds_norm, asc_loadalign)
SIMD_INST_DEFINE_LOAD(vlds_brc_elem, asc_loadalign_brc_elem)
SIMD_INST_DEFINE_LOAD(vlds_upsample, asc_loadalign_upsample)
SIMD_INST_DEFINE_LOAD(vlds_downsample, asc_loadalign_downsample)
SIMD_INST_DEFINE_LOAD(vlds_unpack, asc_loadalign_unpack)
SIMD_INST_DEFINE_LOAD(vlds_unpack4, asc_loadalign_unpack4)
SIMD_INST_DEFINE_LOAD(vlds_brc_datablock, asc_loadalign_brc_datablock)
SIMD_INST_DEFINE_LOAD(vlds_brc_elem2datablock, asc_loadalign_brc_elem2datablock)

#undef SIMD_INST_DEFINE_LOAD

// Predicate-register loads. One entry per asc_loadalign* predicate variant;
// mirrors the vlds_* split, so codegen selects the distribution by name
// instead of threading a `dist` branch through the body.
__simd_callee__ inline vector_bool plds_norm(__ubuf__ uint32_t *src,
                                             int32_t offset) {
  vector_bool dst;
  asc_loadalign(dst, src, offset);
  return dst;
}

__simd_callee__ inline vector_bool plds_upsample(__ubuf__ uint32_t *src,
                                                 int32_t offset) {
  vector_bool dst;
  asc_loadalign_upsample(dst, src, offset);
  return dst;
}

__simd_callee__ inline vector_bool plds_downsample(__ubuf__ uint32_t *src,
                                                   int32_t offset) {
  vector_bool dst;
  asc_loadalign_downsample(dst, src, offset);
  return dst;
}

// Predicate-register stores. One entry per asc_storealign* predicate variant.
__simd_callee__ inline void psts_norm(vector_bool src, __ubuf__ uint32_t *base,
                                      int32_t offset) {
  asc_storealign(base, src, offset);
}

__simd_callee__ inline void psts_pack(vector_bool src, __ubuf__ uint32_t *base,
                                      int32_t offset) {
  asc_storealign_pack(base, src, offset);
}

// Dual-dest memory load (ASC DIST_DINTLV_B16). Prefer ::vld(dst0,dst1,...)
// which wraps ::vlds on dav-3510; matches asc_loadalign_v2_impl.h.
template <typename T, typename Dist>
__simd_callee__ inline vec_pair<vec_t<T>> vld_x2(__ubuf__ T *src, Dist dist) {
  vec_pair<vec_t<T>> dst;
  asc_loadalign_deintlv(dst.v0, dst.v1, src);
  return dst;
}

template <typename T, typename Offset, typename Dist>
__simd_callee__ inline vec_pair<vec_t<T>> vld_x2(__ubuf__ T *src, Offset offset,
                                                 Dist dist) {
  vec_pair<vec_t<T>> dst;
  asc_loadalign_deintlv(dst.v0, dst.v1, src, offset);
  return dst;
}

template <typename T>
__simd_callee__ inline vec_t<T> vgatherb(__ubuf__ T *base, vector_u32 idx) {
  vec_t<T> dst;
  asc_gather_datablock(dst, base, idx);
  return dst;
}

template <typename T>
__simd_callee__ inline vec_t<T> vgatherb(__ubuf__ T *base, vector_u32 idx,
                                         vector_bool mask) {
  vec_t<T> dst;
  asc_gather_datablock(dst, base, idx, mask);
  return dst;
}

template <typename T, typename IdxVec>
__simd_callee__ inline vec_t<T> vgather2(__ubuf__ T *base, IdxVec idx,
                                         vector_bool mask) {
  vec_t<T> dst;
  asc_gather(dst, base, idx, mask);
  return dst;
}

template <typename IdxVec>
__simd_callee__ inline widen_vec_t<vec_t<int8_t>>
vgather2(__ubuf__ int8_t *base, IdxVec idx, vector_bool mask) {
  widen_vec_t<vec_t<int8_t>> dst;
  asc_gather(dst, base, idx, mask);
  return dst;
}

template <typename IdxVec>
__simd_callee__ inline widen_vec_t<vec_t<uint8_t>>
vgather2(__ubuf__ uint8_t *base, IdxVec idx, vector_bool mask) {
  widen_vec_t<vec_t<uint8_t>> dst;
  asc_gather(dst, base, idx, mask);
  return dst;
}

template <typename T, typename IdxVec>
__simd_callee__ inline void vscatter(vec_t<T> data, __ubuf__ T *base,
                                     IdxVec idx, vector_bool mask) {
  asc_scatter(base, data, idx, mask);
}

__simd_callee__ inline vector_bool pand(vector_bool src_0, vector_bool src_1,
                                        vector_bool mask) {
  vector_bool dst;
  asc_and(dst, src_0, src_1, mask);
  return dst;
}

__simd_callee__ inline vector_bool por(vector_bool src_0, vector_bool src_1,
                                       vector_bool mask) {
  vector_bool dst;
  asc_or(dst, src_0, src_1, mask);
  return dst;
}

__simd_callee__ inline vector_bool pxor(vector_bool src_0, vector_bool src_1,
                                        vector_bool mask) {
  vector_bool dst;
  asc_xor(dst, src_0, src_1, mask);
  return dst;
}

__simd_callee__ inline vector_bool pnot(vector_bool src, vector_bool mask) {
  vector_bool dst;
  asc_not(dst, src, mask);
  return dst;
}

__simd_callee__ inline vector_bool psel(vector_bool src_0, vector_bool src_1,
                                        vector_bool mask) {
  vector_bool dst;
  asc_select(dst, src_0, src_1, mask);
  return dst;
}

// Keep conversion dispatch on the CCE intrinsic to cover its full type matrix.
// The trailing controls vary by dtype: part/round, optional rs/part, then mode.
template <typename U, typename SrcVec, typename... Controls>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool mask,
                                     Controls... controls) {
  vec_t<U> dst;
  ::vcvt(dst, src, mask, controls...);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<SrcVec> vdintlv(SrcVec src_0, SrcVec src_1) {
  vec_pair<SrcVec> dst;
  asc_deintlv(dst.v0, dst.v1, src_0, src_1);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<SrcVec> vintlv(SrcVec src_0, SrcVec src_1) {
  vec_pair<SrcVec> dst;
  asc_intlv(dst.v0, dst.v1, src_0, src_1);
  return dst;
}

// -- Binary arithmetic
// ----------------------------------------------------------

// Vector and scalar RHS operands share the same zeroing/merging adapter.
// LegalizeSimdMerging supplies dst only for MODE_MERGING calls.
#define SIMD_INST_DEFINE_BINARY(Op, CApi)                                      \
  template <typename SrcVec, typename RHS, typename Mode>                      \
  __simd_callee__ inline SrcVec Op(SrcVec src_0, RHS src_1, vector_bool mask,  \
                                   Mode) {                                     \
    SrcVec dst;                                                                \
    CApi(dst, src_0, src_1, mask);                                             \
    return dst;                                                                \
  }                                                                            \
  template <typename SrcVec, typename RHS, typename Mode>                      \
  __simd_callee__ inline SrcVec Op(SrcVec &dst, SrcVec src_0, RHS src_1,       \
                                   vector_bool mask, Mode) {                   \
    SrcVec result;                                                             \
    CApi(result, src_0, src_1, mask);                                          \
    return merge_masked(dst, result, mask);                                    \
  }

SIMD_INST_DEFINE_BINARY(vadd, asc_add)
SIMD_INST_DEFINE_BINARY(vsub, asc_sub)
SIMD_INST_DEFINE_BINARY(vmul, asc_mul)
SIMD_INST_DEFINE_BINARY(vdiv, asc_div)
SIMD_INST_DEFINE_BINARY(vmax, asc_max)
SIMD_INST_DEFINE_BINARY(vmin, asc_min)
SIMD_INST_DEFINE_BINARY(vand, asc_and)
SIMD_INST_DEFINE_BINARY(vor, asc_or)
SIMD_INST_DEFINE_BINARY(vxor, asc_xor)
SIMD_INST_DEFINE_BINARY(vshl, asc_shiftleft)
SIMD_INST_DEFINE_BINARY(vshr, asc_shiftright)
SIMD_INST_DEFINE_BINARY(vabsdif, asc_abs_sub)

SIMD_INST_DEFINE_BINARY(vadds, asc_add_scalar)
SIMD_INST_DEFINE_BINARY(vmaxs, asc_max_scalar)
SIMD_INST_DEFINE_BINARY(vmins, asc_min_scalar)
SIMD_INST_DEFINE_BINARY(vmuls, asc_mul_scalar)
SIMD_INST_DEFINE_BINARY(vshls, asc_shiftleft_scalar)
SIMD_INST_DEFINE_BINARY(vshrs, asc_shiftright_scalar)

#undef SIMD_INST_DEFINE_BINARY

template <typename SrcVec>
__simd_callee__ inline vec_pair<vector_bool, SrcVec>
vaddc(SrcVec src_0, SrcVec src_1, vector_bool mask) {
  vec_pair<vector_bool, SrcVec> dst;
  asc_add(dst.v0, reinterpret_cast<vector_u32 &>(dst.v1),
          reinterpret_cast<vector_u32 &>(src_0),
          reinterpret_cast<vector_u32 &>(src_1), mask);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<vector_bool, SrcVec>
vsubc(SrcVec src_0, SrcVec src_1, vector_bool mask) {
  vec_pair<vector_bool, SrcVec> dst;
  asc_sub(dst.v0, reinterpret_cast<vector_u32 &>(dst.v1),
          reinterpret_cast<vector_u32 &>(src_0),
          reinterpret_cast<vector_u32 &>(src_1), mask);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<vector_bool, SrcVec>
vaddcs(SrcVec src_0, SrcVec src_1, vector_bool carrysrcp, vector_bool mask) {
  vec_pair<vector_bool, SrcVec> dst;
  asc_addc(dst.v0, reinterpret_cast<vector_u32 &>(dst.v1),
           reinterpret_cast<vector_u32 &>(src_0),
           reinterpret_cast<vector_u32 &>(src_1), carrysrcp, mask);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<vector_bool, SrcVec>
vsubcs(SrcVec src_0, SrcVec src_1, vector_bool carrysrcp, vector_bool mask) {
  vec_pair<vector_bool, SrcVec> dst;
  asc_subc(dst.v0, reinterpret_cast<vector_u32 &>(dst.v1),
           reinterpret_cast<vector_u32 &>(src_0),
           reinterpret_cast<vector_u32 &>(src_1), carrysrcp, mask);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<SrcVec> vmull(SrcVec src_0, SrcVec src_1,
                                              vector_bool mask) {
  vec_pair<SrcVec> dst;
  asc_mull(dst.v0, dst.v1, src_0, src_1, mask);
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
  SrcVec old_dst = *dst;
  SrcVec result = old_dst;
  asc_axpy(result, src, scalar, mask);
  if constexpr (mode == MODE_ZEROING) {
    *dst = result;
  } else {
    asc_select(*dst, result, old_dst, mask);
  }
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline void vaxpy(SrcVec &dst, SrcVec src, ScalarT scalar,
                                  vector_bool mask, Mode mode) {
  vaxpy(&dst, src, scalar, mask, mode);
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
  asc_duplicate_scalar((vector_u32 &)regNegZero, signBitNum, mask);

  vector_f32 z;
  asc_div(z, src0, src1, mask);

  vector_u32 infNan;
  asc_or(infNan, (vector_u32 &)z, (vector_u32 &)regNegZero, mask);

  vector_f32 tmpDst = z;

  vector_bool zeroCmp;
  asc_eq_scalar(zeroCmp, z, 0.0f, mask);
  vector_bool infNanCmp;
  asc_ge_scalar(infNanCmp, infNan, infNanBound, mask);
  asc_or(infNanCmp, infNanCmp, zeroCmp, mask);

  vector_f32 y;
  asc_mul_scalar(y, src1, -1.0f, mask);
  vector_f32 r = src0;
  ::vmula(r, z, y, mask, mode);

  vector_f32 rPre, rNext, zPre, zNext;
  asc_add_scalar((vector_s32 &)zPre, (vector_s32 &)z, -1, mask);
  asc_add_scalar((vector_s32 &)zNext, (vector_s32 &)z, 1, mask);

  rPre = src0;
  rNext = src0;
  ::vmula(rPre, zPre, y, mask, mode);
  ::vmula(rNext, zNext, y, mask, mode);

  asc_abs(r, r, mask);
  asc_abs(rPre, rPre, mask);
  asc_abs(rNext, rNext, mask);

  vector_bool cmpMaskReg;
  asc_lt(cmpMaskReg, r, rPre, mask);
  asc_select(r, r, rPre, cmpMaskReg);
  asc_select(z, z, zPre, cmpMaskReg);

  asc_lt(cmpMaskReg, rNext, r, mask);
  asc_select(z, zNext, z, cmpMaskReg);

  vector_f32 dst;
  asc_select(dst, tmpDst, z, infNanCmp);
  return dst;
}

template <typename Mode>
__simd_callee__ inline vector_f32
vdiv_0ulp_ftz_true(vector_f32 &dst, vector_f32 src0, vector_f32 src1,
                   vector_bool mask, Mode mode) {
  (void)mode;
  vector_f32 result = vdiv_0ulp_ftz_true(src0, src1, mask, MODE_ZEROING);
  return merge_masked(dst, result, mask);
}

// -- Unary
// ----------------------------------------------------------------------

#define SIMD_INST_DEFINE_UNARY(Op, CApi)                                       \
  template <typename SrcVec, typename Mode>                                    \
  __simd_callee__ inline SrcVec Op(SrcVec src, vector_bool mask, Mode) {       \
    SrcVec dst;                                                                \
    CApi(dst, src, mask);                                                      \
    return dst;                                                                \
  }                                                                            \
  template <typename SrcVec, typename Mode>                                    \
  __simd_callee__ inline SrcVec Op(SrcVec &dst, SrcVec src, vector_bool mask,  \
                                   Mode) {                                     \
    SrcVec result;                                                             \
    CApi(result, src, mask);                                                   \
    return merge_masked(dst, result, mask);                                    \
  }

SIMD_INST_DEFINE_UNARY(vln, asc_ln)
SIMD_INST_DEFINE_UNARY(vsqrt, asc_sqrt)
SIMD_INST_DEFINE_UNARY(vabs, asc_abs)
SIMD_INST_DEFINE_UNARY(vneg, asc_neg)
SIMD_INST_DEFINE_UNARY(vrelu, asc_relu)
SIMD_INST_DEFINE_UNARY(vnot, asc_not)
SIMD_INST_DEFINE_UNARY(vexp, asc_exp)

#undef SIMD_INST_DEFINE_UNARY

template <typename SrcVec, typename ScalarT>
__simd_callee__ inline SrcVec vlrelu(SrcVec src, ScalarT alpha,
                                     vector_bool mask) {
  SrcVec dst;
  asc_leakyrelu(dst, src, alpha, mask);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline SrcVec vprelu(SrcVec src_0, SrcVec src_1,
                                     vector_bool mask) {
  SrcVec dst;
  asc_prelu(dst, src_0, src_1, mask);
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
  asc_exp(z, src, mask);                    // SFU initial value
  asc_le_scalar(m, z, kMaxSubnormal, mask); // output would be subnormal?
  asc_mul_scalar(half, src, 0.5f, mask);
  asc_exp(t, half, mask); // e^(x/2): stays in normal range
  asc_mul(t, t, t, mask);
  SrcVec dst;
  asc_select(dst, t, z, m);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vexp_1ulp_ftz_false(SrcVec &dst, SrcVec src,
                                                  vector_bool mask, Mode mode) {
  (void)mode;
  SrcVec result = vexp_1ulp_ftz_false(src, mask, MODE_ZEROING);
  return merge_masked(dst, result, mask);
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
  asc_ln(z, src, mask);                      // hardware path
  asc_lt_scalar(sub, src, kMinNormal, mask); // subnormal magnitude
  asc_gt_scalar(pos, src, 0.0f, mask);       // positive only
  asc_and(m, sub, pos, mask);
  asc_mul_scalar(scaled, src, 8388608.0f, mask); // 2^23
  asc_ln(t, scaled, mask);
  asc_add_scalar(t, t, -kLn2p23, mask);
  SrcVec dst;
  asc_select(dst, t, z, m);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vln_1ulp_ftz_false(SrcVec &dst, SrcVec src,
                                                 vector_bool mask, Mode mode) {
  (void)mode;
  SrcVec result = vln_1ulp_ftz_false(src, mask, MODE_ZEROING);
  return merge_masked(dst, result, mask);
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
  asc_lt_scalar(p, src, kOne, mask); // scale inputs < 1 up
  asc_mul_scalar(scaled, src, kScaleUp, mask);
  asc_select(b, scaled, src, p);
  asc_duplicate_scalar(one, kOne, mask);
  asc_sqrt(tmp, b, mask);
  asc_div(x, one, tmp, mask); // 1/sqrt(b) initial value
  asc_mul_scalar(tmp, x, -kOne, mask);
  asc_mul(err, x, b, mask);
  ::vmula(one, err, tmp, mask, mode); // first Newton step
  asc_mul_scalar(tmp, x, kHalf, mask);
  ::vmula(x, one, tmp, mask, mode);
  asc_mul(res, x, b, mask);
  asc_mul_scalar(tmp, res, -kOne, mask);
  err = b;
  ::vmula(err, res, tmp, mask, mode); // second residual correction
  asc_mul_scalar(tmp, x, kHalf, mask);
  asc_madd(tmp, err, res, mask);
  asc_mul_scalar(scaled, tmp, kScaleDn, mask);
  asc_select(tmp, scaled, tmp, p);          // unscale only scaled inputs
  asc_eq_scalar(isZero, src, 0.0f, mask);   // +-0 pass through
  asc_eq_scalar(isInf, src, kPosInf, mask); // +inf pass through
  asc_or(special, isZero, isInf, mask);
  SrcVec dst;
  asc_select(dst, src, tmp, special); // special ? src : tmp
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec
vsqrt_0ulp_ftz_false(SrcVec &dst, SrcVec src, vector_bool mask, Mode mode) {
  (void)mode;
  SrcVec result = vsqrt_0ulp_ftz_false(src, mask, MODE_ZEROING);
  return merge_masked(dst, result, mask);
}

// -- Cross-lane reductions
// ------------------------------------------------------
template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcpadd(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  asc_pair_reduce_sum(dst, src, mask);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline widen_vec_t<SrcVec> vcadd(SrcVec src, vector_bool mask,
                                                 Mode mode) {
  widen_vec_t<SrcVec> dst;
  asc_reduce_sum(dst, src, mask);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline widen_vec_t<SrcVec>
vcadd(widen_vec_t<SrcVec> &dst, SrcVec src, vector_bool mask, Mode mode) {
  (void)mode;
  using DstVec = widen_vec_t<SrcVec>;
  DstVec result;
  asc_reduce_sum(result, src, mask);
  return merge_masked(dst, result, reduction_result_mask<DstVec>(PAT_VL1));
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcmax(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  asc_reduce_max(dst, src, mask);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcmax(SrcVec &dst, SrcVec src, vector_bool mask,
                                    Mode mode) {
  (void)mode;
  SrcVec result;
  asc_reduce_max(result, src, mask);
  return merge_masked(dst, result, reduction_result_mask<SrcVec>(PAT_VL2));
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcmin(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  asc_reduce_min(dst, src, mask);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcmin(SrcVec &dst, SrcVec src, vector_bool mask,
                                    Mode mode) {
  (void)mode;
  SrcVec result;
  asc_reduce_min(result, src, mask);
  return merge_masked(dst, result, reduction_result_mask<SrcVec>(PAT_VL2));
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcgadd(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  asc_reduce_sum_datablock(dst, src, mask);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcgmax(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  asc_reduce_max_datablock(dst, src, mask);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcgmin(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  asc_reduce_min_datablock(dst, src, mask);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsqz(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  asc_squeeze(dst, src, mask);
  return dst;
}

// vusqz: per-lane exclusive prefix count of mask; merge-mode needs pre-zeroed
// Vd via vdup (a `{}` init lowers to a BUILD_VECTOR the backend rejects).
template <typename T> __simd_callee__ inline vec_t<T> vusqz(vector_bool mask) {
  vec_t<T> dst;
  asc_duplicate_scalar(dst, static_cast<T>(0), mask);
  asc_unsqueeze(dst, mask);
  return dst;
}

__simd_callee__ inline vector_bool update_mask_b8(uint32_t value) {
  return asc_update_mask_b8(value);
}

__simd_callee__ inline vector_bool update_mask_b16(uint32_t value) {
  return asc_update_mask_b16(value);
}

__simd_callee__ inline vector_bool update_mask_b32(uint32_t value) {
  return asc_update_mask_b32(value);
}

template <typename Part>
__simd_callee__ inline vector_bool ppack(vector_bool src, Part part) {
  vector_bool dst;
  if constexpr (part == HIGHER) {
    asc_pack_to_high(dst, src);
  } else {
    asc_pack_to_low(dst, src);
  }
  return dst;
}

template <typename Part>
__simd_callee__ inline vector_bool punpack(vector_bool src, Part part) {
  vector_bool dst;
  if constexpr (part == HIGHER) {
    asc_unpack_upper(dst, src);
  } else {
    asc_unpack_lower(dst, src);
  }
  return dst;
}

#define TL_SIMD_PINTLV_IMPL(WIDTH)                                             \
  __simd_callee__ inline vec_pair<vector_bool> pintlv_##WIDTH(                 \
      vector_bool src_0, vector_bool src_1) {                                  \
    vec_pair<vector_bool> dst;                                                 \
    asc_intlv_##WIDTH(dst.v0, dst.v1, src_0, src_1);                           \
    return dst;                                                                \
  }                                                                            \
  __simd_callee__ inline vec_pair<vector_bool> pdintlv_##WIDTH(                \
      vector_bool src_0, vector_bool src_1) {                                  \
    vec_pair<vector_bool> dst;                                                 \
    asc_deintlv_##WIDTH(dst.v0, dst.v1, src_0, src_1);                         \
    return dst;                                                                \
  }

TL_SIMD_PINTLV_IMPL(b8)
TL_SIMD_PINTLV_IMPL(b16)
TL_SIMD_PINTLV_IMPL(b32)
#undef TL_SIMD_PINTLV_IMPL

template <typename Bin>
__simd_callee__ inline void dhistv2(vector_u16 *dst, vector_u8 src,
                                    vector_bool mask, Bin bin) {
  if constexpr (bin == Bin_N0) {
    asc_frequency_histogram_bin0(*dst, src, mask);
  } else {
    asc_frequency_histogram_bin1(*dst, src, mask);
  }
}

template <typename Bin>
__simd_callee__ inline void chistv2(vector_u16 *dst, vector_u8 src,
                                    vector_bool mask, Bin bin) {
  if constexpr (bin == Bin_N0) {
    asc_cumulative_histogram_bin0(*dst, src, mask);
  } else {
    asc_cumulative_histogram_bin1(*dst, src, mask);
  }
}

// -- Index ramp / compare
// ------------------------------------------------------------

// index ramp: T = int32_t / float / ...; returns dst[lane] = index (+/-) lane
template <typename T, typename Order>
__simd_callee__ inline vec_t<T> vci(T index, Order order) {
  vec_t<T> dst;
  if constexpr (order == INC_ORDER) {
    asc_arange(dst, index);
  } else {
    asc_arange_descend(dst, index);
  }
  return dst;
}

#define SIMD_INST_DEFINE_COMPARE(OP)                                           \
  template <typename SrcVec>                                                   \
  __simd_callee__ inline vector_bool vcmp_##OP(SrcVec src_0, SrcVec src_1,     \
                                               vector_bool mask) {             \
    vector_bool dst;                                                           \
    asc_##OP(dst, src_0, src_1, mask);                                         \
    return dst;                                                                \
  }                                                                            \
  template <typename SrcVec, typename ScalarT>                                 \
  __simd_callee__ inline vector_bool vcmps_##OP(SrcVec src, ScalarT scalar,    \
                                                vector_bool mask) {            \
    vector_bool dst;                                                           \
    asc_##OP##_scalar(dst, src, scalar, mask);                                 \
    return dst;                                                                \
  }
SIMD_INST_DEFINE_COMPARE(eq)
SIMD_INST_DEFINE_COMPARE(ne)
SIMD_INST_DEFINE_COMPARE(gt)
SIMD_INST_DEFINE_COMPARE(ge)
SIMD_INST_DEFINE_COMPARE(lt)
SIMD_INST_DEFINE_COMPARE(le)
#undef SIMD_INST_DEFINE_COMPARE

// -- Broadcast
// ------------------------------------------------------------------

// scalar broadcast: T = float / half / int32_t / ...
template <typename T, typename Mode>
__simd_callee__ inline vec_t<T> vdup(T src, vector_bool mask, Mode mode) {
  vec_t<T> dst;
  asc_duplicate_scalar(dst, src, mask);
  return dst;
}

template <typename T, typename Mode>
__simd_callee__ inline vec_t<T> vdup(vec_t<T> &dst, T src, vector_bool mask,
                                     Mode mode) {
  (void)mode;
  vec_t<T> result;
  asc_duplicate_scalar(result, src, mask);
  return merge_masked(dst, result, mask);
}

template <typename SrcVec, typename Pos, typename Mode>
__simd_callee__ inline SrcVec vdupv(SrcVec src, vector_bool mask, Pos pos,
                                    Mode mode) {
  SrcVec dst;
  if constexpr (pos == POS_LOWEST) {
    asc_duplicate(dst, src, mask);
  } else {
    asc_duplicate_highest(dst, src, mask);
  }
  return dst;
}

template <typename SrcVec, typename Pos, typename Mode>
__simd_callee__ inline SrcVec vdupv(SrcVec &dst, SrcVec src, vector_bool mask,
                                    Pos pos, Mode mode) {
  (void)mode;
  SrcVec result = vdupv(src, mask, pos, MODE_ZEROING);
  return merge_masked(dst, result, mask);
}

// -- Select
// ---------------------------------------------------------------------

template <typename SrcVec>
__simd_callee__ inline SrcVec vsel(SrcVec src_0, SrcVec src_1,
                                   vector_bool mask) {
  SrcVec dst;
  asc_select(dst, src_0, src_1, mask);
  return dst;
}

template <typename SrcVec, typename IdxVec>
__simd_callee__ inline SrcVec vselr(SrcVec src, IdxVec idx) {
  SrcVec dst;
  asc_gather(dst, src, idx);
  return dst;
}

// -- Exponential difference
// -----------------------------------------------------

template <typename SrcVec>
__simd_callee__ inline SrcVec vexpdif(SrcVec src_0, SrcVec src_1,
                                      vector_bool mask) {
  SrcVec dst;
  asc_exp_sub(dst, src_0, src_1, mask);
  return dst;
}

template <typename U, typename SrcVec, typename Part>
__simd_callee__ inline vec_t<U> vpack(SrcVec src, Part part) {
  vec_t<U> dst;
  if constexpr (part == HIGHER) {
    asc_pack_to_high(dst, src);
  } else {
    asc_pack_to_low(dst, src);
  }
  return dst;
}

template <typename SrcVec, typename Part>
__simd_callee__ inline widen_vec_t<SrcVec> vunpack(SrcVec src, Part part) {
  widen_vec_t<SrcVec> dst;
  if constexpr (part == HIGHER) {
    asc_unpack_upper(dst, src);
  } else {
    asc_unpack_lower(dst, src);
  }
  return dst;
}

template <typename T>
__simd_callee__ inline void vsstb(vec_t<T> src, __ubuf__ T *base,
                                  int32_t stride, vector_bool mask) {
  asc_storealign(base, src, static_cast<uint16_t>(stride >> 16),
                 static_cast<uint16_t>(stride), mask);
}

template <typename T, typename Post>
__simd_callee__ inline __ubuf__ T *vsstb(vec_t<T> src, __ubuf__ T *base,
                                         int32_t stride, vector_bool mask,
                                         Post post) {
  asc_storealign_postupdate(base, src, static_cast<uint16_t>(stride >> 16),
                            static_cast<uint16_t>(stride), mask);
  return base;
}

// Register stores. One entry per asc_storealign* distribution; the element
// width (B8/B16/B32) is carried by T, so the width-suffixed dist names
// collapse onto one function per family. Mirrors the vlds_* / plds_* split.
template <typename T>
__simd_callee__ inline void vsts_norm(vec_t<T> data, __ubuf__ T *base,
                                      int32_t offset, vector_bool mask) {
  asc_storealign(base, data, offset, mask);
}

template <typename T>
__simd_callee__ inline void vsts_1st(vec_t<T> data, __ubuf__ T *base,
                                     int32_t offset, vector_bool mask) {
  asc_storealign_1st(base, data, offset);
}

template <typename T>
__simd_callee__ inline void vsts_pack_b16(vec_t<T> data, __ubuf__ T *base,
                                          int32_t offset, vector_bool mask) {
  asc_storealign_pack((__ubuf__ uint16_t *)base,
                      reinterpret_cast<vector_uint16_t &>(data), offset, mask);
}

template <typename T>
__simd_callee__ inline void vsts_pack_b32(vec_t<T> data, __ubuf__ T *base,
                                          int32_t offset, vector_bool mask) {
  asc_storealign_pack((__ubuf__ uint32_t *)base,
                      reinterpret_cast<vector_uint32_t &>(data), offset, mask);
}

template <typename T>
__simd_callee__ inline void vsts_pack_quarter(vec_t<T> data, __ubuf__ T *base,
                                              int32_t offset,
                                              vector_bool mask) {
  if constexpr (std::is_same<T, int32_t>::value ||
                std::is_same<T, uint32_t>::value ||
                std::is_same<T, float>::value) {
    asc_storealign_pack_quarter(base, data, offset, mask);
  } else if constexpr (std::is_same<T, fp8_e4_t>::value ||
                       std::is_same<T, fp8_e5_t>::value ||
                       std::is_same<T, float4_e2m1x2_t>::value ||
                       std::is_same<T, float4_e1m2x2_t>::value ||
                       std::is_same<T, uint8_t>::value ||
                       std::is_same<T, int8_t>::value) {
    asc_storealign_pack_quarter((__ubuf__ uint32_t *)base, (vector_u32 &)data,
                                offset, mask);
  }
}

template <typename T>
__simd_callee__ inline void vsts_intlv(vec_t<T> data, __ubuf__ T *base,
                                       int32_t offset, vector_bool mask) {
  asc_storealign_intlv(base, data, offset, mask);
}

template <typename T> __simd_callee__ inline void mem_bar(T mem_type) {
  ::asc_mem_bar(mem_type);
}

} // namespace simd_inst
